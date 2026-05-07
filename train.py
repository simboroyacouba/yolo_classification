"""
Entrainement YOLO dual - nadir (panneau_solaire) / oblique (batiments)

Architecture duale :
  - Mode nadir   : classe panneau_solaire   (instances_nadir.json)
  - Mode oblique : classes batiment_*        (instances_oblique.json)
  - Mode dual    : nadir puis oblique sequentiellement

Ameliorations vs version initiale :
  - Augmentations : flipH, flipV, ColorJitter, CosineAnnealingLR
  - Oversampling  : pilote par --aug CLASS:COEFF (remplace dict hardcode)
  - Staged training : geler backbone N epochs puis degeler avec LR/5
  - AP@50 par classe a chaque epoch et apres entrainement
  - CBAM : injection post-load (backbone pre-entraine conserve)

Usage :
  python train.py --mode simple
  python train.py --mode attention --cbam-reduction 16
  python train.py --mode optimize --n-trials 20
  python train.py --mode dual --aug panneau_solaire:3 batiment_peint:2
  python train.py --mode dual --freeze-epochs 5
"""

import os
import copy
import json
import yaml
import shutil
import argparse
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time
import gc
import torch
import torch.nn as nn
import csv
import warnings
warnings.filterwarnings("ignore")

from attention import CBAM

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONSTANTES
# =============================================================================

MODE_CLASSES = {
    "nadir":   ["panneau_solaire"],
    "oblique": [
        "batiment_peint", "batiment_non_enduit", "batiment_enduit",
        "menuiserie_metallique",
    ],
}

OPTUNA_CONFIG = {
    "n_trials":           20,
    "n_epochs_per_trial": 5,
    "study_name":         "yolo_cadastral",
    "output_dir":         "./optuna_output",
}


# =============================================================================
# CBAM — ENREGISTREMENT ET CONSTRUCTION DU YAML CUSTOM
# =============================================================================

class _CBAMWrapper(nn.Module):
    """
    Enveloppe une couche YOLO avec un bloc CBAM.
    Preserve les attributs .i, .f, .type necessaires au forward pass de DetectionModel.
    Classe au niveau module (picklable).
    """
    def __init__(self, layer, cbam):
        super().__init__()
        self.layer = layer
        self.cbam  = cbam
        self.i    = getattr(layer, "i",    -1)
        self.f    = getattr(layer, "f",    -1)
        self.type = getattr(layer, "type", type(layer).__name__) + "+CBAM"
        self.np   = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        return self.cbam(self.layer(x))


def _inject_cbam_post_load(model, model_version, model_size,
                            cbam_reduction=16, cbam_kernel_size=7):
    """
    Injecte CBAM directement dans le modele charge, sans modifier le YAML.
    Strategie : enveloppe les couches backbone cibles avec _CBAMWrapper.
    Les poids pre-entraines sont conserves ; seuls les blocs CBAM sont nouveaux.
    """
    import ultralytics

    ver = model_version.replace("yolo", "")
    pkg = Path(ultralytics.__file__).parent

    candidates = [
        pkg / "cfg" / "models" / ver / f"{model_version}.yaml",
        pkg / "cfg" / "models" / f"v{ver}" / f"{model_version}.yaml",
    ]
    base_yaml_path = next((p for p in candidates if p.exists()), None)
    if base_yaml_path is None:
        print("   Avertissement : YAML de base non trouve — CBAM non injecte")
        return model

    with open(base_yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    insert_at = _find_cbam_positions(cfg.get("backbone", []))
    if not insert_at:
        print("   Avertissement : aucun point d'insertion detecte — CBAM non injecte")
        return model

    print(f"   Points d'insertion CBAM : couches backbone {insert_at}")

    layers   = model.model.model
    injected = 0

    for idx in insert_at:
        target_key = None
        for key, m in layers._modules.items():
            if key == str(idx) or getattr(m, "i", None) == idx:
                target_key = key
                break

        if target_key is None:
            print(f"   Avertissement : layer {idx} non trouve dans le modele")
            continue

        layer = layers._modules[target_key]

        c_out = None
        for m in reversed(list(layer.modules())):
            if isinstance(m, nn.BatchNorm2d):
                c_out = m.num_features
                break
            if isinstance(m, nn.Conv2d):
                c_out = m.out_channels
                break

        if c_out is None:
            print(f"   Avertissement : canaux non detectes pour layer {idx}")
            continue

        cbam_block = CBAM(c_out, reduction=cbam_reduction, kernel_size=cbam_kernel_size)
        layers._modules[target_key] = _CBAMWrapper(layer, cbam_block)
        injected += 1

    print(f"   {injected} blocs CBAM injectes (r={cbam_reduction}, k={cbam_kernel_size})")
    return model


def _get_layer_out_channels(layer, width_multiple, max_channels):
    args = layer[3] if len(layer) > 3 else []
    if isinstance(args, list) and args and isinstance(args[0], int):
        return min(max(round(args[0] * width_multiple), 1), max_channels)
    return None


def _find_cbam_positions(backbone):
    """Detecte les indices backbone apres lesquels inserer un CBAM."""
    positions = []
    attention_blocks = {"C3k2", "C2f", "C3", "C3x", "C3TR", "BottleneckCSP"}

    for i, layer in enumerate(backbone):
        module = layer[2] if len(layer) > 2 else ""
        if module == "SPPF":
            positions.append(i)
        elif module in attention_blocks and i + 1 < len(backbone):
            nxt        = backbone[i + 1]
            nxt_module = nxt[2] if len(nxt) > 2 else ""
            nxt_args   = nxt[3] if len(nxt) > 3 else []
            if nxt_module == "Conv" and isinstance(nxt_args, list) and len(nxt_args) >= 3 and nxt_args[2] == 2:
                positions.append(i)

    return positions


def build_cbam_yaml(model_version, model_size, output_dir):
    """Genere un YAML Ultralytics avec des blocs CBAM dans le backbone."""
    import ultralytics

    pkg  = Path(ultralytics.__file__).parent
    ver  = model_version.replace("yolo", "")

    candidates = [
        pkg / "cfg" / "models" / ver / f"{model_version}.yaml",
        pkg / "cfg" / "models" / f"v{ver}" / f"{model_version}.yaml",
        pkg / "models" / ver / f"{model_version}.yaml",
    ]
    base_yaml_path = next((p for p in candidates if p.exists()), None)
    if base_yaml_path is None:
        raise FileNotFoundError(
            f"YAML de base introuvable pour {model_version}.\n"
            f"Chemins essayes : {[str(c) for c in candidates]}"
        )

    with open(base_yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    scales      = cfg.get("scales", {})
    scale       = scales.get(model_size, [1.0, 1.0, 1024])
    depth_mult  = scale[0]
    width_mult  = scale[1]
    max_ch      = scale[2] if len(scale) > 2 else 1024

    backbone  = cfg.get("backbone", [])
    head      = cfg.get("head",     [])
    n_bb_orig = len(backbone)

    insert_at = _find_cbam_positions(backbone)
    if not insert_at:
        print("   Avertissement — aucun point d'insertion CBAM detecte dans le backbone.")
        return None

    print(f"   Points d'insertion CBAM : couches backbone {insert_at}")

    new_backbone = []
    cbam_shifts  = {}
    shift        = 0

    for orig_idx, layer in enumerate(backbone):
        new_abs = orig_idx + shift
        new_backbone.append(layer)

        if orig_idx in insert_at:
            c_out = _get_layer_out_channels(layer, width_mult, max_ch) or 256
            new_backbone.append([-1, 1, "CBAM", [c_out]])
            cbam_shifts[orig_idx] = new_abs + 1
            shift += 1
        else:
            cbam_shifts[orig_idx] = new_abs

    n_inserted = shift

    def _update_ref(r):
        if not isinstance(r, int) or r < 0:
            return r
        if r < n_bb_orig:
            return cbam_shifts[r]
        return r + n_inserted

    new_head = []
    for layer in head:
        f     = layer[0]
        new_f = [_update_ref(x) for x in f] if isinstance(f, list) else _update_ref(f)
        new_head.append([new_f] + layer[1:])

    cfg["backbone"] = new_backbone
    cfg["head"]     = new_head

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_version}{model_size}_cbam.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=None, allow_unicode=True)

    print(f"   YAML CBAM genere : {out_path}  ({n_inserted} blocs CBAM)")
    return out_path


def _load_pretrained_partial(model, pretrained_path):
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    src  = ckpt.get("model", ckpt)
    if hasattr(src, "state_dict"):
        src = src.float().state_dict()

    dst      = model.model.state_dict()
    filtered = {k: v for k, v in src.items()
                if k in dst and v.shape == dst[k].shape}
    dst.update(filtered)
    model.model.load_state_dict(dst)
    print(f"   Poids pre-entraines : {len(filtered)}/{len(dst)} couches chargees")


# =============================================================================
# PARSE AUG COEFFICIENTS
# =============================================================================

def parse_aug_coeffs(aug_args, classes):
    """
    Analyse --aug CLASS:COEFF avec correspondance partielle insensible a la casse.
    Retourne un dict {class_name: coeff} pour les classes connues.
    """
    coeffs = {}
    for entry in (aug_args or []):
        if ':' not in entry:
            print(f"   [WARN] --aug '{entry}' ignore (format attendu CLASS:COEFF)")
            continue
        raw_name, raw_coeff = entry.rsplit(':', 1)
        try:
            coeff = int(raw_coeff)
        except ValueError:
            print(f"   [WARN] --aug '{entry}' ignore (coefficient non entier)")
            continue
        if coeff < 1:
            print(f"   [WARN] --aug '{entry}' ignore (coefficient < 1)")
            continue
        raw_lower = raw_name.lower()
        matched = [c for c in classes if raw_lower in c.lower()]
        if not matched:
            print(f"   [WARN] --aug '{raw_name}' ne correspond a aucune classe connue")
            continue
        if len(matched) > 1:
            print(f"   [WARN] --aug '{raw_name}' ambigu ({matched}), ignore")
            continue
        coeffs[matched[0]] = coeff
    return coeffs


# =============================================================================
# CONFIGURATION
# =============================================================================

def build_config(args):
    base_annotations = os.getenv(
        "DETECTION_DATASET_ANNOTATIONS_FILE",
        "../dataset1/annotations/instances_default.json",
    )
    ann_dir = os.path.dirname(os.path.abspath(base_annotations))

    annotations_file = args.annotations_file or base_annotations
    classes_file     = args.classes_file or os.getenv("CLASSES_FILE", "classes.yaml")
    base_output      = os.getenv("OUTPUT_DIR", "./output")
    output_dir       = args.output_dir or base_output

    return {
        "mode":             args.mode,
        "images_dir":       args.images_dir or os.getenv(
                                "DETECTION_DATASET_IMAGES_DIR",
                                "../dataset1/images/default"
                            ),
        "annotations_file": annotations_file,
        "ann_dir":          ann_dir,
        "classes_file":     classes_file,
        "output_dir":       output_dir,
        "model_version":    os.getenv("YOLO_VERSION", "yolo26"),
        "model_size":       os.getenv("YOLO_SIZE", "l"),
        "num_epochs":       int(os.getenv("NUM_EPOCHS", "25")),
        "batch_size":       int(os.getenv("BATCH_SIZE", "2")),
        "learning_rate":    float(os.getenv("LEARNING_RATE", "0.005")),
        "momentum":         float(os.getenv("MOMENTUM",     "0.8784")),
        "weight_decay":     float(os.getenv("WEIGHT_DECAY", "0.00001")),
        "image_size":       int(os.getenv("IMAGE_SIZE", "640")),
        "train_split":      float(os.getenv("TRAIN_SPLIT", "0.70")),
        "val_split":        float(os.getenv("VAL_SPLIT", "0.20")),
        "test_split":       float(os.getenv("TEST_SPLIT", "0.10")),
        "save_every":       5,
        "freeze_epochs":    args.freeze_epochs,
        "use_attention":    getattr(args, "attention", "none"),
        "classes":          None,
    }


def _build_sub_config(base_config, mode_label, mode_classes):
    """Construit un sous-config nadir ou oblique depuis le config de base."""
    ann_dir = base_config["ann_dir"]
    cfg = copy.deepcopy(base_config)
    cfg["mode"] = mode_label
    if mode_label == "nadir":
        cfg["annotations_file"] = os.path.join(ann_dir, "instances_nadir.json")
        cfg["output_dir"]       = os.path.join(base_config["output_dir"], "nadir")
    elif mode_label == "oblique":
        cfg["annotations_file"] = os.path.join(ann_dir, "instances_oblique.json")
        cfg["output_dir"]       = os.path.join(base_config["output_dir"], "oblique")
    cfg["classes"] = mode_classes
    return cfg


# =============================================================================
# CLASSES
# =============================================================================

def load_classes(yaml_path, mode_classes=None):
    """Charger et filtrer les classes depuis le YAML."""

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Fichier classes introuvable : {yaml_path}\n"
            f"Conseil : lancez d'abord  python split_dataset.py"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_classes = [c for c in data.get("classes", []) if c != "__background__"]

    if mode_classes is not None:
        classes = [c for c in all_classes if c in mode_classes]
        if not classes:
            classes = list(mode_classes)
    else:
        classes = all_classes

    print(f"Classes depuis {yaml_path}:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")

    return classes


# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def coco_to_yolo_bbox(bbox, img_width, img_height):
    x, y, w, h = bbox
    cx = max(0.0, min(1.0, (x + w / 2) / img_width))
    cy = max(0.0, min(1.0, (y + h / 2) / img_height))
    nw = max(0.0, min(1.0, w / img_width))
    nh = max(0.0, min(1.0, h / img_height))
    return [cx, cy, nw, nh]


# =============================================================================
# SPLIT DATASET
# =============================================================================

def stratified_split(coco, train_split, val_split, test_split, seed=42):
    np.random.seed(seed)

    all_ids = [img_id for img_id in coco.imgs if coco.getAnnIds(imgIds=img_id)]
    np.random.shuffle(all_ids)

    n_total = len(all_ids)
    n_train = int(n_total * train_split)
    n_val   = int(n_total * val_split)
    n_test  = n_total - n_train - n_val

    if n_test < 1 and n_total > 2:
        n_test  = max(1, int(n_total * 0.10))
        n_train = n_total - n_val - n_test

    print(f"\n   Split des images (total : {n_total}) :")
    print(f"      Train : {n_train}  ({n_train / n_total * 100:.1f}%)")
    print(f"      Val   : {n_val}   ({n_val   / n_total * 100:.1f}%)")
    print(f"      Test  : {n_test}  ({n_test  / n_total * 100:.1f}%)")

    train_ids = all_ids[:n_train]
    val_ids   = all_ids[n_train:n_train + n_val]
    test_ids  = all_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


# =============================================================================
# OVERSAMPLING (file-based, pour YOLO)
# =============================================================================

def oversample_train_set(images_dir, labels_dir, classes, weights_map):
    """
    Dupliquer les images d'entrainement pour les classes sous-representees.
    weights_map : {class_name: facteur}
    """
    weight_by_idx = {
        idx: weights_map[cls_name]
        for idx, cls_name in enumerate(classes)
        if cls_name in weights_map
    }

    if not weight_by_idx:
        return 0

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    n_copies = 0

    for lbl_file in sorted(Path(labels_dir).glob("*.txt")):
        if "_os" in lbl_file.stem:
            continue

        max_w = 1
        try:
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        max_w  = max(max_w, weight_by_idx.get(cls_id, 1))
        except Exception:
            continue

        if max_w <= 1:
            continue

        img_file = None
        for ext in IMG_EXTS:
            p = Path(images_dir) / (lbl_file.stem + ext)
            if p.exists():
                img_file = p
                break

        if img_file is None:
            continue

        for k in range(1, max_w):
            stem = f"{lbl_file.stem}_os{k}"
            shutil.copy2(lbl_file, Path(labels_dir) / f"{stem}.txt")
            shutil.copy2(img_file, Path(images_dir) / f"{stem}{img_file.suffix}")
            n_copies += 1

    return n_copies


# =============================================================================
# PREPARATION DU DATASET YOLO
# =============================================================================

def prepare_yolo_dataset(images_dir, annotations_file, output_dir, classes,
                         train_split, val_split, test_split, mode="all",
                         aug_coeffs=None):
    """
    Convertir COCO -> format YOLO avec split train/val/test et oversampling
    pilote par aug_coeffs (remplace OVERSAMPLE_WEIGHTS_OBLIQUE).
    """
    print("Preparation du dataset YOLO...")

    base_output = os.path.abspath(output_dir)
    dataset_dir = os.path.join(base_output, "dataset")

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    dirs = {
        "train_img": os.path.join(dataset_dir, "images",  "train"),
        "val_img":   os.path.join(dataset_dir, "images",  "val"),
        "test_img":  os.path.join(dataset_dir, "images",  "test"),
        "train_lbl": os.path.join(dataset_dir, "labels",  "train"),
        "val_lbl":   os.path.join(dataset_dir, "labels",  "val"),
        "test_lbl":  os.path.join(dataset_dir, "labels",  "test"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    coco = COCO(annotations_file)

    cat_ids        = coco.getCatIds()
    cat_name_to_id = {coco.cats[cid]["name"]: cid for cid in cat_ids}
    valid_cat_ids  = [cat_name_to_id[c] for c in classes if c in cat_name_to_id]
    cat_mapping    = {cat_id: idx for idx, cat_id in enumerate(valid_cat_ids)}

    print(f"   Mode '{mode}' — classes : {[coco.cats[c]['name'] for c in valid_cat_ids]}")

    train_ids, val_ids, test_ids = stratified_split(
        coco, train_split, val_split, test_split, seed=42
    )

    splits_map = {
        "train": (train_ids, dirs["train_img"], dirs["train_lbl"]),
        "val":   (val_ids,   dirs["val_img"],   dirs["val_lbl"]),
        "test":  (test_ids,  dirs["test_img"],  dirs["test_lbl"]),
    }

    stats = {
        "train": 0, "val": 0, "test": 0, "annotations": 0,
        "per_class": {c: 0 for c in classes},
        "ann_per_split": {"train": 0, "val": 0, "test": 0},
        "per_class_per_split": {
            split: {c: 0 for c in classes}
            for split in ("train", "val", "test")
        },
    }

    for split_name, (img_ids, img_dir, lbl_dir) in splits_map.items():
        for img_id in img_ids:
            img_info = coco.imgs[img_id]
            src      = os.path.join(images_dir, img_info["file_name"])
            if not os.path.exists(src):
                continue

            shutil.copy2(src, os.path.join(img_dir, img_info["file_name"]))

            anns     = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            lbl_path = os.path.join(
                lbl_dir,
                os.path.splitext(img_info["file_name"])[0] + ".txt",
            )

            with open(lbl_path, "w") as f:
                for ann in anns:
                    if ann.get("iscrowd", 0):
                        continue
                    class_id = cat_mapping.get(ann["category_id"])
                    if class_id is None:
                        continue
                    bbox = ann.get("bbox")
                    if not bbox or bbox[2] <= 0 or bbox[3] <= 0:
                        continue

                    yolo_bbox = coco_to_yolo_bbox(
                        bbox, img_info["width"], img_info["height"]
                    )
                    f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in yolo_bbox)}\n")
                    stats["annotations"] += 1
                    stats["ann_per_split"][split_name] += 1
                    if class_id < len(classes):
                        stats["per_class"][classes[class_id]] += 1
                        stats["per_class_per_split"][split_name][classes[class_id]] += 1

            stats[split_name] += 1

    # Oversampling via aug_coeffs
    if aug_coeffs:
        print("\n   Oversampling (aug_coeffs) :")
        for cls_name, w in aug_coeffs.items():
            if cls_name in classes:
                print(f"      {cls_name:<30} x{w}")
        n_copies = oversample_train_set(
            dirs["train_img"], dirs["train_lbl"],
            classes, aug_coeffs,
        )
        stats["oversampling_copies"] = n_copies
        print(f"   + {n_copies} copies ajoutees au train set")

        post_os = {c: 0 for c in classes}
        for lbl_file in Path(dirs["train_lbl"]).glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(parts[0]) < len(classes):
                        post_os[classes[int(parts[0])]] += 1
        stats["per_class_after_oversampling"] = post_os

    # dataset.yaml
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(
            {
                "path":  os.path.abspath(dataset_dir),
                "train": "images/train",
                "val":   "images/val",
                "test":  "images/test",
                "names": {i: name for i, name in enumerate(classes)},
            },
            f,
            default_flow_style=False,
        )

    with open(os.path.join(dataset_dir, "test_info.json"), "w") as f:
        json.dump(
            {
                "mode":            mode,
                "test_images_dir": os.path.abspath(dirs["test_img"]),
                "test_labels_dir": os.path.abspath(dirs["test_lbl"]),
                "num_test_images": stats["test"],
                "classes":         classes,
            },
            f,
            indent=2,
        )

    post_os = stats.get("per_class_after_oversampling")
    has_os  = post_os is not None

    total_imgs    = stats["train"] + stats["val"] + stats["test"]
    train_imgs_os = len(list(Path(dirs["train_img"]).glob("*.*")))
    print(f"\n   {'':=<60}")
    print(f"   REPARTITION DU DATASET")
    print(f"   {'':=<60}")
    if has_os:
        print(f"   {'Split':<10} {'Images':>8}  {'Apres OS':>9}  {'Annotations':>12}")
        print(f"   {'-'*45}")
        print(f"   {'Train':<10} {stats['train']:>8}  {train_imgs_os:>9}  {stats['ann_per_split']['train']:>12}")
    else:
        print(f"   {'Split':<10} {'Images':>8}  {'Annotations':>12}")
        print(f"   {'-'*34}")
        print(f"   {'Train':<10} {stats['train']:>8}  {stats['ann_per_split']['train']:>12}")
    print(f"   {'Val':<10} {stats['val']:>8}  {stats['val']:>9}  {stats['ann_per_split']['val']:>12}" if has_os
          else f"   {'Val':<10} {stats['val']:>8}  {stats['ann_per_split']['val']:>12}")
    print(f"   {'Test':<10} {stats['test']:>8}  {stats['test']:>9}  {stats['ann_per_split']['test']:>12}" if has_os
          else f"   {'Test':<10} {stats['test']:>8}  {stats['ann_per_split']['test']:>12}")
    print(f"   {'Total':<10} {total_imgs:>8}  {'':>9}  {stats['annotations']:>12}" if has_os
          else f"   {'Total':<10} {total_imgs:>8}  {stats['annotations']:>12}")

    print(f"\n   Augmentations actives :")
    print(f"      flipud  = 0.5   (flip vertical)")
    print(f"      hsv_h   = 0.05  (teinte)")
    print(f"      hsv_s   = 0.162 (saturation)")
    print(f"      hsv_v   = 0.113 (luminosite)")
    print(f"      fliplr  = 0.0   (flip horizontal desactive)")
    print(f"   Dataset : {dataset_dir}")

    return yaml_path, stats


# =============================================================================
# CALLBACKS
# =============================================================================

def make_staged_training_callback(freeze_epochs):
    def on_train_epoch_start(trainer):
        if trainer.epoch == 0:
            for name, v in trainer.model.named_parameters():
                if "cbam" in name.lower():
                    v.requires_grad = True

        if trainer.epoch == freeze_epochs:
            for _, v in trainer.model.named_parameters():
                v.requires_grad = True
            for pg in trainer.optimizer.param_groups:
                lr = pg.get("initial_lr", pg["lr"])
                pg["lr"]         = lr / 5
                pg["initial_lr"] = lr / 5
            print(f"\n   Backbone degele (epoch {freeze_epochs + 1}) | LR divise par 5")

    return on_train_epoch_start


def make_per_class_ap_callback():
    def on_fit_epoch_end(trainer):
        if not hasattr(trainer, "validator"):
            return
        validator = trainer.validator
        if not hasattr(validator, "metrics"):
            return

        box = getattr(validator.metrics, "box", None)
        if box is None:
            return

        ap_class_index = getattr(box, "ap_class_index", None)
        ap_matrix      = getattr(box, "ap", None)

        if ap_class_index is None or ap_matrix is None:
            return
        if len(ap_class_index) == 0:
            return

        ap50  = ap_matrix[:, 0] if ap_matrix.ndim == 2 else ap_matrix
        names = trainer.data.get("names", {})

        print(f"\n   AP@50 par classe (epoch {trainer.epoch + 1}) :")
        for idx, ap in zip(ap_class_index, ap50):
            name = names.get(int(idx), f"class_{idx}")
            print(f"      {name:<25} {ap:.4f}")

    return on_fit_epoch_end


# =============================================================================
# AFFICHAGE AP PAR CLASSE
# =============================================================================

def display_per_class_ap(model, yaml_path, split="val"):
    print(f"\n   AP@50 par classe — validation finale (split={split}) :")

    try:
        val_results = model.val(data=yaml_path, split=split, verbose=False)
        box   = val_results.box
        names = val_results.names

        ap50_per_class = None
        ap_class_index = getattr(box, "ap_class_index", None)

        if hasattr(box, "ap") and box.ap is not None and box.ap.ndim == 2:
            ap50_per_class = box.ap[:, 0]
        elif hasattr(box, "maps") and box.maps is not None:
            ap50_per_class = box.maps

        if ap50_per_class is not None and ap_class_index is not None and len(ap_class_index) > 0:
            print(f"   {'Classe':<30} {'AP@50':>8}")
            print(f"   {'-'*42}")
            for idx, ap in zip(ap_class_index, ap50_per_class):
                name = names.get(int(idx), f"class_{idx}")
                print(f"   {name:<30} {ap:>8.4f}")
            print(f"   {'-'*42}")

        print(f"   {'mAP@50':<30} {box.map50:>8.4f}")
        print(f"   {'mAP@50:95':<30} {box.map:>8.4f}")

    except Exception as e:
        print(f"   Avertissement — AP par classe indisponible : {e}")


# =============================================================================
# COURBES D'ENTRAINEMENT
# =============================================================================

def plot_training_curves(history, train_dir, mode):
    if not history.get("mAP50"):
        return

    epochs = range(1, len(history["mAP50"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Courbes d'entrainement YOLO — Mode {mode.upper()}", fontsize=13)

    def safe_sum(a, b, c):
        return [x + y + z for x, y, z in zip(a, b, c)]

    train_loss = safe_sum(
        history["train_box_loss"], history["train_cls_loss"], history["train_dfl_loss"]
    )
    val_loss = safe_sum(
        history["val_box_loss"], history["val_cls_loss"], history["val_dfl_loss"]
    )
    axes[0, 0].plot(epochs, train_loss, "b-", label="Train")
    axes[0, 0].plot(epochs, val_loss,   "r-", label="Val")
    axes[0, 0].set_title("Loss totale"); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["mAP50"],    "g-", label="mAP@50")
    axes[0, 1].plot(epochs, history["mAP50_95"], "b-", label="mAP@50:95")
    axes[0, 1].set_title("mAP"); axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].set_ylim(0, 1)

    axes[1, 0].plot(epochs, history["precision"], "g-", label="Precision")
    axes[1, 0].plot(epochs, history["recall"],    "b-", label="Recall")
    axes[1, 0].set_title("Precision / Recall"); axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_ylim(0, 1)

    axes[1, 1].plot(epochs, history["train_box_loss"], label="Box")
    axes[1, 1].plot(epochs, history["train_cls_loss"], label="Cls")
    axes[1, 1].plot(epochs, history["train_dfl_loss"], label="DFL")
    axes[1, 1].set_title("Composantes Loss (train)")
    axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(train_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   Courbes : {out}")


# =============================================================================
# ENTRAINEMENT PRINCIPAL
# =============================================================================

def _train_single(config, aug_coeffs, mode_label, training_mode,
                  cbam_reduction=16, cbam_kernel_size=7):
    """
    Lance un entraînement YOLO complet.
    training_mode: 'simple' | 'attention'
    """
    classes = config["classes"]

    print("=" * 70)
    print(f"   YOLO ({config['model_version']}{config['model_size']}) — Mode : {mode_label.upper()} [{training_mode}]")
    print("=" * 70)
    print(f"\n   Configuration :")
    print(f"   Mode          : {mode_label}")
    print(f"   Images        : {config['images_dir']}")
    print(f"   Annotations   : {config['annotations_file']}")
    print(f"   Classes       : {classes}")
    print(f"   Modele        : {config['model_version']}{config['model_size']}")
    print(f"   Epochs        : {config['num_epochs']}  Batch : {config['batch_size']}  LR : {config['learning_rate']}")
    print(f"   Freeze epochs : {config['freeze_epochs']}")
    print(f"   Attention     : {training_mode}")
    if aug_coeffs:
        print(f"   Oversampling  : {aug_coeffs}")

    if not os.path.exists(config["annotations_file"]):
        raise FileNotFoundError(
            f"Annotations introuvables : {config['annotations_file']}"
        )

    os.makedirs(config["output_dir"], exist_ok=True)

    yaml_path, dataset_stats = prepare_yolo_dataset(
        config["images_dir"],
        config["annotations_file"],
        config["output_dir"],
        classes,
        config["train_split"],
        config["val_split"],
        config["test_split"],
        mode=mode_label,
        aug_coeffs=aug_coeffs,
    )

    model_name = f"{config['model_version']}{config['model_size']}.pt"
    use_cbam   = training_mode == "attention"

    gc.collect()

    if use_cbam:
        print(f"\n   Chargement avec CBAM (post-load) : {model_name}")
        model = YOLO(model_name)
        _inject_cbam_post_load(model, config["model_version"], config["model_size"],
                               cbam_reduction, cbam_kernel_size)
    else:
        print(f"\n   Chargement : {model_name}")
        model = YOLO(model_name)

    model.add_callback("on_fit_epoch_end", make_per_class_ap_callback())

    freeze_n_layers = 0
    if config["freeze_epochs"] > 0:
        freeze_n_layers = 10
        model.add_callback(
            "on_train_epoch_start",
            make_staged_training_callback(config["freeze_epochs"]),
        )
        print(
            f"\n   Staged training : {freeze_n_layers} couches gelees "
            f"pour les {config['freeze_epochs']} premieres epochs"
        )

    print("\n" + "=" * 70)
    print(f"   ENTRAINEMENT — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    start_time = time.time()

    results = model.train(
        data=yaml_path,
        epochs=config["num_epochs"],
        batch=config["batch_size"],
        imgsz=config["image_size"],
        optimizer="SGD",
        lr0=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        cos_lr=True,
        fliplr=0.0,
        flipud=0.5,
        degrees=0.0,
        hsv_h=0.05,
        hsv_s=0.162,
        hsv_v=0.113,
        mosaic=0.0,
        mixup=0.0,
        freeze=freeze_n_layers if config["freeze_epochs"] > 0 else 0,
        project=mode_label,
        name="train",
        seed=42,
        verbose=True,
        save=True,
        save_period=config["save_every"],
        plots=True,
        cache=False,
        workers=0,
    )

    total_time = time.time() - start_time

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_dir   = str(results.save_dir)
    weights_dir = os.path.join(train_dir, "weights")

    print(f"\n   Dossier YOLO : {train_dir}")

    display_per_class_ap(model, yaml_path, split="val")

    history = {
        "mAP50": [], "mAP50_95": [], "precision": [], "recall": [],
        "train_box_loss": [], "train_cls_loss": [], "train_dfl_loss": [],
        "val_box_loss":   [], "val_cls_loss":   [], "val_dfl_loss":   [],
    }

    results_csv = os.path.join(train_dir, "results.csv")
    if os.path.exists(results_csv):
        with open(results_csv, "r") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}
                for key, col in [
                    ("mAP50",          "metrics/mAP50(B)"),
                    ("mAP50_95",       "metrics/mAP50-95(B)"),
                    ("precision",      "metrics/precision(B)"),
                    ("recall",         "metrics/recall(B)"),
                    ("train_box_loss", "train/box_loss"),
                    ("train_cls_loss", "train/cls_loss"),
                    ("train_dfl_loss", "train/dfl_loss"),
                    ("val_box_loss",   "val/box_loss"),
                    ("val_cls_loss",   "val/cls_loss"),
                    ("val_dfl_loss",   "val/dfl_loss"),
                ]:
                    try:
                        history[key].append(float(row.get(col, 0) or 0))
                    except Exception:
                        pass

    history["time_stats"] = {
        "total_time":               total_time,
        "total_time_formatted":     format_time(total_time),
        "avg_epoch_time_formatted": format_time(total_time / max(config["num_epochs"], 1)),
    }
    history["config"]        = {
        k: v for k, v in config.items()
        if isinstance(v, (str, int, float, bool, list))
    }
    history["dataset_stats"] = dataset_stats

    with open(os.path.join(train_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=str)

    print("\n   Copie des modeles :")
    for src, dst in [("best.pt", "best_model.pt"), ("last.pt", "final_model.pt")]:
        src_p = os.path.join(weights_dir, src)
        dst_p = os.path.join(train_dir, dst)
        if os.path.exists(src_p):
            shutil.copy2(src_p, dst_p)
            print(f"   {dst} ({os.path.getsize(dst_p) / 1024 / 1024:.1f} MB)")
        else:
            print(f"   Avertissement — {src} non trouve dans {weights_dir}")

    model_info = {
        "mode":          mode_label,
        "training_mode": training_mode,
        "train_dir":     train_dir,
        "best_model":    os.path.join(train_dir, "best_model.pt"),
        "final_model":   os.path.join(train_dir, "final_model.pt"),
        "dataset_yaml":  yaml_path,
        "classes":       classes,
        "image_size":    config["image_size"],
        "trained_at":    datetime.now().isoformat(),
    }
    info_path = os.path.join(config["output_dir"], f"model_info_{mode_label}.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    plot_training_curves(history, train_dir, mode_label)

    best = {k: max(history[k]) if history[k] else 0
            for k in ["mAP50", "mAP50_95", "precision", "recall"]}

    print("\n" + "=" * 70)
    print(f"   TERMINE — {mode_label.upper()} [{training_mode}]")
    print("=" * 70)
    print(f"   mAP@50      : {best['mAP50']:.4f} ({best['mAP50'] * 100:.2f}%)")
    print(f"   mAP@50:95   : {best['mAP50_95']:.4f}")
    print(f"   Precision   : {best['precision']:.4f}")
    print(f"   Recall      : {best['recall']:.4f}")
    print(f"   Temps       : {format_time(total_time)}")
    print(f"   Modeles     : {train_dir}")
    print("=" * 70)

    with open(os.path.join(train_dir, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"YOLO ({config['model_version']}{config['model_size']}) — Mode {mode_label} [{training_mode}]\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Mode           : {mode_label}\n")
        f.write(f"Training mode  : {training_mode}\n")
        f.write(f"Dataset        : {config['images_dir']}\n")
        f.write(f"Classes        : {classes}\n")
        f.write(f"Epochs         : {config['num_epochs']}  Batch : {config['batch_size']}\n")
        f.write(f"LR schedule    : CosineAnnealingLR (cos_lr=True)\n")
        f.write(f"Freeze         : {config['freeze_epochs']} epochs\n\n")
        f.write(f"mAP@50         : {best['mAP50']:.4f}\n")
        f.write(f"mAP@50:95      : {best['mAP50_95']:.4f}\n")
        f.write(f"Precision      : {best['precision']:.4f}\n")
        f.write(f"Recall         : {best['recall']:.4f}\n")
        f.write(f"Temps          : {format_time(total_time)}\n")
        f.write(f"Chemin         : {train_dir}\n")

    return model, history, train_dir


# =============================================================================
# OPTIMISATION OPTUNA
# =============================================================================

def _run_optimization(config, aug_coeffs, mode_label, cbam_reduction, cbam_kernel_size,
                      n_trials, n_epochs_per_trial):
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

    classes = config["classes"]

    # Préparer le dataset une seule fois
    yaml_path, _ = prepare_yolo_dataset(
        config["images_dir"],
        config["annotations_file"],
        os.path.join(config["output_dir"], "optuna_dataset"),
        classes,
        config["train_split"],
        config["val_split"],
        config["test_split"],
        mode=mode_label,
        aug_coeffs=aug_coeffs,
    )

    model_name = f"{config['model_version']}{config['model_size']}.pt"

    def objective(trial):
        lr           = trial.suggest_float("lr",           1e-4, 1e-1, log=True)
        momentum     = trial.suggest_float("momentum",     0.7,  0.99)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        use_cbam     = trial.suggest_categorical("use_cbam", [True, False])

        model = YOLO(model_name)
        if use_cbam:
            reduction   = trial.suggest_categorical("cbam_reduction",   [8, 16, 32])
            kernel_size = trial.suggest_categorical("cbam_kernel_size",  [3, 5, 7])
            _inject_cbam_post_load(model, config["model_version"], config["model_size"],
                                   reduction, kernel_size)

        try:
            results = model.train(
                data=yaml_path,
                epochs=n_epochs_per_trial,
                batch=config["batch_size"],
                imgsz=config["image_size"],
                optimizer="SGD",
                lr0=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                cos_lr=True,
                fliplr=0.0,
                flipud=0.5,
                hsv_h=0.05,
                hsv_s=0.162,
                hsv_v=0.113,
                mosaic=0.0,
                project=f"{mode_label}/optuna",
                name=f"trial_{trial.number}",
                seed=42,
                verbose=False,
                save=False,
                plots=False,
                cache=False,
                workers=0,
            )
            map50 = getattr(getattr(results, "box", None), "map50", 0.0) or 0.0
        except Exception as e:
            print(f"   Trial {trial.number} erreur: {e}")
            map50 = 0.0
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return float(map50)

    output_dir = OPTUNA_CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(),
        pruner=MedianPruner(),
        study_name=f"{OPTUNA_CONFIG['study_name']}_{mode_label}",
    )
    print(f"\n   Optuna: {n_trials} trials × {n_epochs_per_trial} epochs chacun")
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    print(f"\n   Meilleurs hyperparametres: {best}")

    with open(os.path.join(output_dir, f"optuna_best_{mode_label}.json"), 'w') as f:
        json.dump({"best_params": best, "best_value": study.best_value}, f, indent=2)

    return best


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entrainement YOLO — detection des toitures cadastrales",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["simple", "attention", "optimize", "dual",
                 "nadir", "oblique", "all"],  # nadir/oblique/all: alias deprecies
        default="simple",
        help="Mode d'entrainement",
    )
    parser.add_argument("--aug",              nargs="*", default=[],
                        metavar="CLASS:COEFF",
                        help="Coefficients d'oversampling par classe (ex: panneau_solaire:3)")
    parser.add_argument("--cbam-reduction",   type=int, default=16,
                        help="Facteur de reduction CBAM (ChannelAttention)")
    parser.add_argument("--cbam-kernel-size", type=int, default=7,
                        help="Taille du kernel CBAM (SpatialAttention)")
    parser.add_argument("--n-trials",         type=int, default=OPTUNA_CONFIG["n_trials"],
                        help="Nombre de trials Optuna")
    parser.add_argument("--n-epochs-trial",   type=int, default=OPTUNA_CONFIG["n_epochs_per_trial"],
                        help="Epochs par trial Optuna")
    parser.add_argument("--images-dir",       default=None)
    parser.add_argument("--annotations-file", default=None)
    parser.add_argument("--classes-file",     default=None)
    parser.add_argument("--output-dir",       default=None)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help="Nombre d'epochs avec backbone gele (staged training). 0 = desactive.",
    )
    parser.add_argument(
        "--attention",
        choices=["none", "cbam"],
        default="none",
        help="(deprecated) Mecanisme d'attention. Utilisez --mode attention.",
    )
    args = parser.parse_args()

    # Alias deprecies
    mode = args.mode
    if mode == "nadir":
        print("[DEPRECATED] --mode nadir est deprecie. Utilisez --mode dual.")
        args.mode = "simple"
        if not args.annotations_file:
            base = os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE",
                             "../dataset1/annotations/instances_default.json")
            args.annotations_file = os.path.join(
                os.path.dirname(os.path.abspath(base)), "instances_nadir.json"
            )
    elif mode == "oblique":
        print("[DEPRECATED] --mode oblique est deprecie. Utilisez --mode dual.")
        args.mode = "simple"
        if not args.annotations_file:
            base = os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE",
                             "../dataset1/annotations/instances_default.json")
            args.annotations_file = os.path.join(
                os.path.dirname(os.path.abspath(base)), "instances_oblique.json"
            )
    elif mode == "all":
        print("[DEPRECATED] --mode all est deprecie. Utilisez --mode simple.")
        args.mode = "simple"
    mode = args.mode

    # --attention cbam upgrades to attention mode
    if getattr(args, "attention", "none") == "cbam" and mode == "simple":
        mode = "attention"
        args.mode = "attention"

    config = build_config(args)

    all_classes = (MODE_CLASSES["nadir"] + MODE_CLASSES["oblique"])

    if mode == "dual":
        # Phase 1 — Nadir
        nadir_classes = load_classes(config["classes_file"], MODE_CLASSES["nadir"])
        nadir_cfg     = _build_sub_config(config, "nadir", nadir_classes)
        nadir_aug     = parse_aug_coeffs(args.aug, nadir_classes)
        print(f"\n[DUAL] Phase 1 — Nadir ({nadir_classes})")
        _train_single(nadir_cfg, nadir_aug, "nadir", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)

        # Phase 2 — Oblique
        oblique_classes = load_classes(config["classes_file"], MODE_CLASSES["oblique"])
        oblique_cfg     = _build_sub_config(config, "oblique", oblique_classes)
        oblique_aug     = parse_aug_coeffs(args.aug, oblique_classes)
        print(f"\n[DUAL] Phase 2 — Oblique ({oblique_classes})")
        _train_single(oblique_cfg, oblique_aug, "oblique", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)

    elif mode == "optimize":
        config["classes"] = load_classes(config["classes_file"], all_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        best_params = _run_optimization(
            config, aug_coeffs, "all",
            args.cbam_reduction, args.cbam_kernel_size,
            args.n_trials, args.n_epochs_trial,
        )
        use_cbam = best_params.get("use_cbam", False)
        config["learning_rate"] = best_params.get("lr",           config["learning_rate"])
        config["momentum"]      = best_params.get("momentum",     config["momentum"])
        config["weight_decay"]  = best_params.get("weight_decay", config["weight_decay"])
        cbam_r = best_params.get("cbam_reduction",   args.cbam_reduction)
        cbam_k = best_params.get("cbam_kernel_size",  args.cbam_kernel_size)
        training_mode = "attention" if use_cbam else "simple"
        _train_single(config, aug_coeffs, "all", training_mode, cbam_r, cbam_k)

    elif mode == "attention":
        config["classes"] = load_classes(config["classes_file"], all_classes)
        aug_coeffs        = parse_aug_coeffs(args.aug, config["classes"])
        _train_single(config, aug_coeffs, "all", "attention",
                      args.cbam_reduction, args.cbam_kernel_size)

    else:  # simple (+ anciens alias redirigés)
        if config["classes"] is None:
            config["classes"] = load_classes(config["classes_file"], all_classes)
        aug_coeffs = parse_aug_coeffs(args.aug, config["classes"])
        _train_single(config, aug_coeffs, "all", "simple",
                      args.cbam_reduction, args.cbam_kernel_size)


if __name__ == "__main__":
    main()
