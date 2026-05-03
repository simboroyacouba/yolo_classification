"""
Entrainement YOLO dual - nadir (panneau_solaire) / oblique (batiments)

Architecture duale :
  - Modele nadir   : entraine sur Production_*.png, classe panneau_solaire
  - Modele oblique : entraine sur Snapshot_*.jpg,   classes batiment_*

Ameliorations vs version initiale :
  - Augmentations : flipH, flipV, ColorJitter, CosineAnnealingLR
  - Oversampling  : batiment_peint x5, batiment_enduit x3, batiment_non_enduit x2
  - Staged training : geler backbone N epochs puis degeler avec LR/5
  - AP@50 par classe a chaque epoch et apres entrainement

Usage :
  python train.py --mode nadir
  python train.py --mode oblique
  python train.py --mode oblique --freeze-epochs 5
  python train.py --mode all
  python train.py --mode oblique --freeze-epochs 5 --images-dir ../dataset1/images/default
"""

import os
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

# Classes selon le mode
MODE_CLASSES = {
    "nadir":   ["panneau_solaire"],
    "oblique": [
        "batiment_peint", "batiment_non_enduit", "batiment_enduit",
        "menuiserie_metallique", "menuiserie_aluminium",
    ],
    "all":     [
        "panneau_solaire",
        "batiment_peint", "batiment_non_enduit", "batiment_enduit",
        "menuiserie_metallique", "menuiserie_aluminium",
    ],
}

# Poids d'oversampling pour le mode oblique
# panneau_solaire retire du modele oblique (P=0.40, gere par le modele nadir)
OVERSAMPLE_WEIGHTS_OBLIQUE = {
    "batiment_peint":        4,
    "batiment_enduit":       1,
    "batiment_non_enduit":   2,
    "menuiserie_metallique": 1,
    "menuiserie_aluminium":  10,
}


# =============================================================================
# CBAM — ENREGISTREMENT ET CONSTRUCTION DU YAML CUSTOM
# =============================================================================

class _CBAMWrapper(nn.Module):
    """
    Enveloppe une couche YOLO avec un bloc CBAM.
    Preserve les attributs .i, .f, .type necessaires au forward pass de DetectionModel.
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


def _inject_cbam_post_load(model, model_version, model_size):
    """
    Injecte CBAM directement dans le modele charge, sans modifier le YAML
    ni dependance au parseur Ultralytics (contournement du KeyError globals()).

    Strategie : enveloppe les couches backbone cibles avec _CBAMWrapper.
    Les poids pre-entraines sont deja charges (YOLO(model_name)).
    Seuls les blocs CBAM sont initialises aleatoirement.
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
        # Trouver la cle dans _modules (peut etre str(idx) ou indexe par .i)
        target_key = None
        for key, m in layers._modules.items():
            if key == str(idx) or getattr(m, "i", None) == idx:
                target_key = key
                break

        if target_key is None:
            print(f"   Avertissement : layer {idx} non trouve dans le modele")
            continue

        layer = layers._modules[target_key]

        # Detecter les canaux de sortie depuis les modules internes
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

        cbam_block = CBAM(c_out)
        layers._modules[target_key] = _CBAMWrapper(layer, cbam_block)
        injected += 1

    print(f"   {injected} blocs CBAM injectes (poids backbone pre-entraines conserves)")
    return model


def _get_layer_out_channels(layer, width_multiple, max_channels):
    """Retourne le nombre de canaux de sortie d'une couche YAML."""
    args = layer[3] if len(layer) > 3 else []
    if isinstance(args, list) and args and isinstance(args[0], int):
        return min(max(round(args[0] * width_multiple), 1), max_channels)
    return None


def _find_cbam_positions(backbone):
    """
    Detecte les indices de backbone apres lesquels inserer un CBAM.
    Regles :
      - Apres chaque bloc C3k2 / C2f / C3 suivi d'un Conv stride-2 (fin de niveau P)
      - Apres le bloc SPPF (fin du backbone)
    """
    positions = []
    attention_blocks = {"C3k2", "C2f", "C3", "C3x", "C3TR", "BottleneckCSP"}

    for i, layer in enumerate(backbone):
        module = layer[2] if len(layer) > 2 else ""
        if module == "SPPF":
            positions.append(i)
        elif module in attention_blocks and i + 1 < len(backbone):
            nxt = backbone[i + 1]
            nxt_module = nxt[2] if len(nxt) > 2 else ""
            nxt_args   = nxt[3] if len(nxt) > 3 else []
            # Conv stride-2 = transition vers un niveau plus profond
            if nxt_module == "Conv" and isinstance(nxt_args, list) and len(nxt_args) >= 3 and nxt_args[2] == 2:
                positions.append(i)

    return positions


def build_cbam_yaml(model_version, model_size, output_dir):
    """
    Genere un YAML Ultralytics avec des blocs CBAM inseres dans le backbone.
    Retourne le chemin du fichier YAML genere.
    """
    import ultralytics

    pkg    = Path(ultralytics.__file__).parent
    ver    = model_version.replace("yolo", "")   # "26" depuis "yolo26"

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

    # Parametres d'echelle
    scales         = cfg.get("scales", {})
    scale          = scales.get(model_size, [1.0, 1.0, 1024])
    depth_mult     = scale[0]
    width_mult     = scale[1]
    max_ch         = scale[2] if len(scale) > 2 else 1024

    backbone = cfg.get("backbone", [])
    head     = cfg.get("head",     [])
    n_bb_orig = len(backbone)

    insert_at = _find_cbam_positions(backbone)
    if not insert_at:
        print("   Avertissement — aucun point d'insertion CBAM detecte dans le backbone.")
        return None

    print(f"   Points d'insertion CBAM : couches backbone {insert_at}")

    # ----- Construire le nouveau backbone -----
    new_backbone = []
    # cbam_shifts[orig_idx] = nouvel indice absolu apres insertion
    # Si CBAM insere apres orig_idx, pointe sur le CBAM (pas sur la couche elle-meme)
    cbam_shifts = {}
    shift = 0

    for orig_idx, layer in enumerate(backbone):
        new_abs = orig_idx + shift
        new_backbone.append(layer)

        if orig_idx in insert_at:
            c_out = _get_layer_out_channels(layer, width_mult, max_ch)
            if c_out is None:
                # Fallback : garder meme nb de canaux que l'entree
                c_out = 256
            new_backbone.append([-1, 1, "CBAM", [c_out]])
            cbam_shifts[orig_idx] = new_abs + 1   # reference = sortie du CBAM
            shift += 1
        else:
            cbam_shifts[orig_idx] = new_abs

    n_inserted = shift   # nombre de CBAM ajoutes

    # ----- Mettre a jour les references dans le head -----
    def _update_ref(r):
        if not isinstance(r, int) or r < 0:
            return r
        if r < n_bb_orig:
            return cbam_shifts[r]      # reference vers le backbone
        return r + n_inserted          # reference vers le head

    new_head = []
    for layer in head:
        f = layer[0]
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
    """
    Charge les poids pre-entraines en ignorant les couches CBAM
    (absent du checkpoint officiel).
    """
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
# CONFIGURATION
# =============================================================================

def build_config(args):
    """Construire la config a partir des args CLI et des variables .env."""

    base_annotations = os.getenv(
        "DETECTION_DATASET_ANNOTATIONS_FILE",
        "../dataset1/annotations/instances_default.json",
    )
    ann_dir = os.path.dirname(os.path.abspath(base_annotations))
    mode    = args.mode

    # Fichier d'annotations selon le mode
    if args.annotations_file:
        annotations_file = args.annotations_file
    elif mode == "nadir":
        annotations_file = os.path.join(ann_dir, "instances_nadir.json")
    elif mode == "oblique":
        annotations_file = os.path.join(ann_dir, "instances_oblique.json")
    else:
        annotations_file = base_annotations

    # Fichier de classes selon le mode
    if args.classes_file:
        classes_file = args.classes_file
    elif mode == "nadir":
        classes_file = "classes_nadir.yaml"
    elif mode == "oblique":
        classes_file = "classes_oblique.yaml"
    else:
        classes_file = os.getenv("CLASSES_FILE", "classes.yaml")

    # Dossier de sortie selon le mode
    if args.output_dir:
        output_dir = args.output_dir
    elif mode in ("nadir", "oblique"):
        output_dir = os.path.join(os.getenv("OUTPUT_DIR", "./output"), mode)
    else:
        output_dir = os.getenv("OUTPUT_DIR", "./output")

    return {
        "mode":             mode,
        "images_dir":       args.images_dir or os.getenv(
                                "DETECTION_DATASET_IMAGES_DIR",
                                "../dataset1/images/default"
                            ),
        "annotations_file": annotations_file,
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
        "classes":          None,  # rempli apres chargement du YAML
    }


# =============================================================================
# CLASSES
# =============================================================================

def load_classes(yaml_path, mode="all"):
    """Charger et filtrer les classes depuis le YAML selon le mode."""

    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Fichier classes introuvable : {yaml_path}\n"
            f"Conseil : lancez d'abord  python split_dataset.py"
        )

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_classes = [c for c in data.get("classes", []) if c != "__background__"]

    # Filtrer selon le mode
    expected = MODE_CLASSES.get(mode, all_classes)
    classes  = [c for c in all_classes if c in expected]

    if not classes:
        # Fallback : utiliser directement les classes attendues
        classes = expected

    print(f"📋 Classes ({mode}) depuis {yaml_path} :")
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
    """Convertir bbox COCO [x,y,w,h] vers YOLO normalise [cx,cy,w,h]."""
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
    """Split global des images (pas de stratification par classe pour YOLO)."""

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
# OVERSAMPLING
# =============================================================================

def oversample_train_set(images_dir, labels_dir, classes, weights_map):
    """
    Dupliquer les images d'entrainement pour les classes sous-representees.

    weights_map : {class_name: facteur}
    Une image est copiee (max_weight - 1) fois supplementaires.
    Retourne le nombre de copies creees.
    """

    # Index : class_idx -> poids
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
        # Ne pas re-oversampler les copies deja creees
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

        # Trouver l'image correspondante
        img_file = None
        for ext in IMG_EXTS:
            p = Path(images_dir) / (lbl_file.stem + ext)
            if p.exists():
                img_file = p
                break

        if img_file is None:
            continue

        # Creer les copies supplementaires
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
                         train_split, val_split, test_split, mode="all"):
    """
    Convertir COCO -> format YOLO avec split train/val/test
    et oversampling optionnel (mode oblique).
    """

    print("📂 Preparation du dataset YOLO...")

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

    # Mapping cat_id COCO -> class_idx YOLO (uniquement les classes du mode)
    cat_ids          = coco.getCatIds()
    cat_name_to_id   = {coco.cats[cid]["name"]: cid for cid in cat_ids}
    valid_cat_ids    = [cat_name_to_id[c] for c in classes if c in cat_name_to_id]
    cat_mapping      = {cat_id: idx for idx, cat_id in enumerate(valid_cat_ids)}

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

    # -------------------------------------------------------------------
    # Oversampling (mode oblique uniquement)
    # -------------------------------------------------------------------

    if mode == "oblique":
        print("\n   Oversampling (oblique) :")
        for cls_name, w in OVERSAMPLE_WEIGHTS_OBLIQUE.items():
            print(f"      {cls_name:<30} x{w}")
        n_copies = oversample_train_set(
            dirs["train_img"], dirs["train_lbl"],
            classes, OVERSAMPLE_WEIGHTS_OBLIQUE,
        )
        stats["oversampling_copies"] = n_copies
        print(f"   + {n_copies} copies ajoutees au train set")

        # Compter les annotations par classe apres oversampling
        post_os = {c: 0 for c in classes}
        for lbl_file in Path(dirs["train_lbl"]).glob("*.txt"):
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and int(parts[0]) < len(classes):
                        post_os[classes[int(parts[0])]] += 1
        stats["per_class_after_oversampling"] = post_os

    # -------------------------------------------------------------------
    # dataset.yaml
    # -------------------------------------------------------------------

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

    # Infos test set pour evaluate_dual.py
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

    post_os    = stats.get("per_class_after_oversampling")
    has_os     = post_os is not None

    # --- Repartition images et annotations par split ---
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
    print(f"   {'-'*45}" if has_os else f"   {'-'*34}")
    print(f"   {'Total':<10} {total_imgs:>8}  {'-':>9}  {stats['annotations']:>12}" if has_os
          else f"   {'Total':<10} {total_imgs:>8}  {stats['annotations']:>12}")

    # --- Distribution par classe par split ---
    if has_os:
        print(f"\n   {'Classe':<30} {'Train':>7}  {'Train OS':>9}  {'Val':>6}  {'Test':>6}")
        print(f"   {'-'*65}")
        for cls_name in stats['per_class']:
            tr    = stats['per_class_per_split']['train'][cls_name]
            tr_os = post_os[cls_name]
            va    = stats['per_class_per_split']['val'][cls_name]
            te    = stats['per_class_per_split']['test'][cls_name]
            print(f"   {cls_name:<30} {tr:>7}  {tr_os:>9}  {va:>6}  {te:>6}")
    else:
        print(f"\n   {'Classe':<30} {'Train':>7}  {'Val':>6}  {'Test':>6}")
        print(f"   {'-'*53}")
        for cls_name in stats['per_class']:
            tr = stats['per_class_per_split']['train'][cls_name]
            va = stats['per_class_per_split']['val'][cls_name]
            te = stats['per_class_per_split']['test'][cls_name]
            print(f"   {cls_name:<30} {tr:>7}  {va:>6}  {te:>6}")

    print(f"\n   Augmentations actives :")
    print(f"      flipud  = 0.5   (flip vertical)")
    print(f"      hsv_h   = 0.05  (teinte)")
    print(f"      hsv_s   = 0.162 (saturation)")
    print(f"      hsv_v   = 0.113 (luminosite)")
    print(f"      fliplr  = 0.0   (flip horizontal desactive — images obliques)")
    print(f"   Dataset      : {dataset_dir}")

    return yaml_path, stats


# =============================================================================
# CALLBACKS
# =============================================================================

def make_staged_training_callback(freeze_epochs):
    """
    Callback qui degele le backbone a l'epoch `freeze_epochs`
    et reduit le LR d'un facteur 5 pour la phase de fine-tuning.
    Les blocs CBAM (poids aleatoires) sont toujours entrainables des le debut.
    """

    def on_train_epoch_start(trainer):
        # CBAM blocks must always be trainable — unfreeze them at epoch 0
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
            print(f"\n🔓 Backbone degele (epoch {freeze_epochs + 1}) | LR divise par 5")

    return on_train_epoch_start


def make_per_class_ap_callback():
    """
    Callback qui affiche l'AP@50 par classe apres chaque validation.
    """

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

        ap50 = ap_matrix[:, 0] if ap_matrix.ndim == 2 else ap_matrix
        names = trainer.data.get("names", {})

        print(f"\n   AP@50 par classe (epoch {trainer.epoch + 1}) :")
        for idx, ap in zip(ap_class_index, ap50):
            name = names.get(int(idx), f"class_{idx}")
            bar  = "█" * int(ap * 20)
            print(f"      {name:<25} {ap:.4f}  {bar}")

    return on_fit_epoch_end


# =============================================================================
# AFFICHAGE AP PAR CLASSE (apres entrainement)
# =============================================================================

def display_per_class_ap(model, yaml_path, split="val"):
    """Lancer une validation et afficher l'AP@50 par classe."""

    print(f"\n📊 AP@50 par classe — validation finale (split={split}) :")

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
                bar  = "█" * int(ap * 20)
                print(f"   {name:<30} {ap:>8.4f}  {bar}")
            print(f"   {'-'*42}")

        print(f"   {'mAP@50':<30} {box.map50:>8.4f}")
        print(f"   {'mAP@50:95':<30} {box.map:>8.4f}")
        print(f"   {'Precision':<30} {box.mp:>8.4f}")
        print(f"   {'Recall':<30} {box.mr:>8.4f}")

    except Exception as e:
        print(f"   Avertissement — AP par classe indisponible : {e}")


# =============================================================================
# COURBES D'ENTRAINEMENT
# =============================================================================

def plot_training_curves(history, train_dir, mode):
    """Tracer et sauvegarder les courbes d'entrainement."""

    if not history.get("mAP50"):
        return

    epochs = range(1, len(history["mAP50"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Courbes d'entrainement YOLO — Mode {mode.upper()}", fontsize=13)

    # Loss totale
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
    axes[0, 0].set_title("Loss totale")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # mAP
    axes[0, 1].plot(epochs, history["mAP50"],    "g-", label="mAP@50")
    axes[0, 1].plot(epochs, history["mAP50_95"], "b-", label="mAP@50:95")
    axes[0, 1].set_title("mAP")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)

    # Precision / Recall
    axes[1, 0].plot(epochs, history["precision"], "g-", label="Precision")
    axes[1, 0].plot(epochs, history["recall"],    "b-", label="Recall")
    axes[1, 0].set_title("Precision / Recall")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)

    # Composantes loss (train)
    axes[1, 1].plot(epochs, history["train_box_loss"], label="Box")
    axes[1, 1].plot(epochs, history["train_cls_loss"], label="Cls")
    axes[1, 1].plot(epochs, history["train_dfl_loss"], label="DFL")
    axes[1, 1].set_title("Composantes Loss (train)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(train_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"   Courbes : {out}")


# =============================================================================
# ENTRAINEMENT
# =============================================================================

def train_yolo(config):
    """Lancer l'entrainement YOLO avec la config donnee."""

    mode    = config["mode"]
    classes = config["classes"]

    print("=" * 70)
    print(f"   YOLO ({config['model_version']}{config['model_size']}) — Mode : {mode.upper()}")
    print("=" * 70)
    print(f"\n📋 Configuration :")
    print(f"   Mode          : {mode}")
    print(f"   Images        : {config['images_dir']}")
    print(f"   Annotations   : {config['annotations_file']}")
    print(f"   Classes       : {classes}")
    print(f"   Modele        : {config['model_version']}{config['model_size']}")
    print(f"   Epochs        : {config['num_epochs']}  Batch : {config['batch_size']}  LR : {config['learning_rate']}")
    print(f"   Freeze epochs : {config['freeze_epochs']}")
    print(f"   LR schedule   : CosineAnnealingLR")
    print(f"   Attention     : {config.get('use_attention', 'none').upper()}")

    if not os.path.exists(config["annotations_file"]):
        raise FileNotFoundError(
            f"Annotations introuvables : {config['annotations_file']}\n"
            f"Lancez d'abord : python split_dataset.py"
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
        mode=mode,
    )

    model_name = f"{config['model_version']}{config['model_size']}.pt"
    use_cbam   = config.get("use_attention", "none") == "cbam"

    gc.collect()

    if use_cbam:
        print(f"\n🧠 Chargement avec CBAM (post-load) : {model_name}")
        model = YOLO(model_name)   # poids pre-entraines charges normalement
        _inject_cbam_post_load(model, config["model_version"], config["model_size"])
    else:
        print(f"\n🧠 Chargement : {model_name}")
        model = YOLO(model_name)

    # -------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------

    # AP@50 par classe a chaque epoch
    model.add_callback("on_fit_epoch_end", make_per_class_ap_callback())

    # Staged training : geler puis degeler
    freeze_n_layers = 0
    if config["freeze_epochs"] > 0:
        freeze_n_layers = 10  # Premiere couches du backbone YOLO
        model.add_callback(
            "on_train_epoch_start",
            make_staged_training_callback(config["freeze_epochs"]),
        )
        print(
            f"\n🔒 Staged training : {freeze_n_layers} couches gelees "
            f"pour les {config['freeze_epochs']} premieres epochs"
        )

    # -------------------------------------------------------------------
    # Entrainement
    # -------------------------------------------------------------------

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
        # ---- CosineAnnealingLR (remplace StepLR trop agressif) ----
        cos_lr=True,
        # ---- Augmentations (calibrees par Optuna) ----
        fliplr=0.0,
        flipud=0.5,
        degrees=0.0,
        hsv_h=0.05,
        hsv_s=0.162,
        hsv_v=0.113,
        mosaic=0.0,
        mixup=0.0,
        # ---- Staged training ----
        freeze=freeze_n_layers if config["freeze_epochs"] > 0 else 0,
        # ---- Divers ----
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

    print(f"\n📁 Dossier YOLO : {train_dir}")

    # -------------------------------------------------------------------
    # AP@50 par classe apres entrainement
    # -------------------------------------------------------------------

    display_per_class_ap(model, yaml_path, split="val")

    # -------------------------------------------------------------------
    # Historique des metriques
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # Copier les modeles
    # -------------------------------------------------------------------

    print("\n📦 Copie des modeles :")
    for src, dst in [("best.pt", "best_model.pt"), ("last.pt", "final_model.pt")]:
        src_p = os.path.join(weights_dir, src)
        dst_p = os.path.join(train_dir, dst)
        if os.path.exists(src_p):
            shutil.copy2(src_p, dst_p)
            print(f"   {dst} ({os.path.getsize(dst_p) / 1024 / 1024:.1f} MB)")
        else:
            print(f"   Avertissement — {src} non trouve dans {weights_dir}")

    # Sauvegarder les infos du modele pour inference_dual.py et evaluate_dual.py
    model_info = {
        "mode":        mode,
        "train_dir":   train_dir,
        "best_model":  os.path.join(train_dir, "best_model.pt"),
        "final_model": os.path.join(train_dir, "final_model.pt"),
        "dataset_yaml": yaml_path,
        "classes":     classes,
        "image_size":  config["image_size"],
        "trained_at":  datetime.now().isoformat(),
    }
    info_path = os.path.join(config["output_dir"], f"model_info_{mode}.json")
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)

    # -------------------------------------------------------------------
    # Courbes d'entrainement
    # -------------------------------------------------------------------

    plot_training_curves(history, train_dir, mode)

    # -------------------------------------------------------------------
    # Rapport final
    # -------------------------------------------------------------------

    best = {k: max(history[k]) if history[k] else 0
            for k in ["mAP50", "mAP50_95", "precision", "recall"]}

    print("\n" + "=" * 70)
    print(f"   TERMINE — {mode.upper()}")
    print("=" * 70)
    print(f"   mAP@50      : {best['mAP50']:.4f} ({best['mAP50'] * 100:.2f}%)")
    print(f"   mAP@50:95   : {best['mAP50_95']:.4f}")
    print(f"   Precision   : {best['precision']:.4f}")
    print(f"   Recall      : {best['recall']:.4f}")
    print(f"   Temps       : {format_time(total_time)}")
    print(f"   Modeles     : {train_dir}")
    print("=" * 70)

    with open(os.path.join(train_dir, "training_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"YOLO ({config['model_version']}{config['model_size']}) — Mode {mode}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset      : {config['images_dir']}\n")
        f.write(f"Mode         : {mode}\n")
        f.write(f"Classes      : {classes}\n")
        f.write(f"Epochs       : {config['num_epochs']}  Batch : {config['batch_size']}\n")
        f.write(f"LR schedule  : CosineAnnealingLR (cos_lr=True)\n")
        f.write(f"Attention    : {config.get('use_attention', 'none').upper()}\n")
        f.write(f"Freeze       : {config['freeze_epochs']} epochs\n\n")
        f.write(f"mAP@50       : {best['mAP50']:.4f}\n")
        f.write(f"mAP@50:95    : {best['mAP50_95']:.4f}\n")
        f.write(f"Precision    : {best['precision']:.4f}\n")
        f.write(f"Recall       : {best['recall']:.4f}\n")
        f.write(f"Temps        : {format_time(total_time)}\n")
        f.write(f"Chemin       : {train_dir}\n")

    return model, history, train_dir


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entrainement YOLO dual — nadir / oblique / all",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["nadir", "oblique", "all"],
        default="all",
        help=(
            "Mode d'entrainement : "
            "nadir (panneau_solaire uniquement), "
            "oblique (batiments uniquement), "
            "all (toutes les classes)"
        ),
    )
    parser.add_argument("--images-dir",       default=None, help="Dossier images du dataset")
    parser.add_argument("--annotations-file", default=None, help="Fichier annotations COCO JSON")
    parser.add_argument("--classes-file",     default=None, help="Fichier classes YAML")
    parser.add_argument("--output-dir",       default=None, help="Dossier de sortie")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help=(
            "Nombre d'epochs avec backbone gele (staged training). "
            "0 = pas de staged training."
        ),
    )
    parser.add_argument(
        "--attention",
        choices=["none", "cbam"],
        default="none",
        help="Mecanisme d'attention a integrer dans le backbone (default: none).",
    )
    args = parser.parse_args()

    config = build_config(args)
    config["classes"] = load_classes(config["classes_file"], mode=config["mode"])

    if not config["classes"]:
        print(f"Erreur : aucune classe pour le mode '{config['mode']}'")
        return

    model, history, train_dir = train_yolo(config)

    print(f"\nProchaines etapes :")
    if config["mode"] == "nadir":
        print(f"   python train.py --mode oblique")
    elif config["mode"] == "oblique":
        print(f"   python train.py --mode nadir")
    print(f"   python evaluate_dual.py")
    print(f"   python inference_dual.py --input ../test")


if __name__ == "__main__":
    main()
