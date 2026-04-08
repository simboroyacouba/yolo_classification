"""
Inference duale YOLO — Routing nadir / oblique par nom de fichier

Routing :
  Production_*.png  ->  modele nadir   (panneau_solaire)
  Snapshot_*.jpg    ->  modele oblique (batiment_peint, batiment_non_enduit, batiment_enduit)

Seuils de confiance par classe :
  panneau_solaire     = 0.60
  batiment_peint      = 0.70
  batiment_non_enduit = 0.55
  batiment_enduit     = 0.65

Usage :
  python inference_dual.py --input ../test
  python inference_dual.py --input ../test --mode both
  python inference_dual.py --nadir-model output/nadir/.../best_model.pt \\
                            --oblique-model output/oblique/.../best_model.pt \\
                            --input ../test
"""

import os
import json
import yaml
import argparse
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import time
import warnings
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONSTANTES
# =============================================================================

NADIR_PREFIX   = "Production_"
OBLIQUE_PREFIX = "Snapshot_"

# Seuils de confiance par classe
CLASS_THRESHOLDS = {
    "panneau_solaire":     0.60,
    "batiment_peint":      0.70,
    "batiment_non_enduit": 0.55,
    "batiment_enduit":     0.65,
}

COLORS = {
    "panneau_solaire":     (255, 0,   0),
    "batiment_peint":      (0,   255, 0),
    "batiment_non_enduit": (0,   0,   255),
    "batiment_enduit":     (255, 165, 0),
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# =============================================================================
# DECOUVERTE AUTOMATIQUE DES MODELES
# =============================================================================

def find_model(mode, output_base="./output"):
    """Trouver automatiquement le meilleur modele pour un mode donne."""

    # 1. Lire model_info_{mode}.json genere par train.py
    info_path = os.path.join(output_base, f"model_info_{mode}.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        best = info.get("best_model")
        if best and os.path.exists(best):
            print(f"   [{mode}] model_info.json -> {best}")
            return best

    # 2. Parcourir output/{mode}/ puis output/
    for base in [os.path.join(output_base, mode), output_base]:
        if not os.path.exists(base):
            continue
        for root, dirs, files in os.walk(base):
            if "weights" in dirs:
                pt = os.path.join(root, "weights", "best.pt")
                if os.path.exists(pt):
                    print(f"   [{mode}] scan -> {pt}")
                    return pt

    # 3. Chercher dans runs/detect/
    if os.path.exists("runs/detect"):
        folders = sorted(
            [f for f in os.listdir("runs/detect") if f.startswith("train")],
            key=lambda x: int(x.replace("train", "") or "0"),
        )
        for folder in reversed(folders):
            pt = f"runs/detect/{folder}/weights/best.pt"
            if os.path.exists(pt):
                print(f"   [{mode}] runs/detect -> {pt}")
                return pt

    return None


def load_classes_from_yaml(yaml_path):
    """Charger les noms de classes depuis un YAML (sans __background__)."""
    if not os.path.exists(yaml_path):
        return []
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [c for c in data.get("classes", []) if c != "__background__"]


# =============================================================================
# ROUTING
# =============================================================================

def detect_image_type(filename):
    """Determiner si une image est nadir (Production_*) ou oblique (Snapshot_*)."""
    name = Path(filename).name
    if name.startswith(NADIR_PREFIX):
        return "nadir"
    if name.startswith(OBLIQUE_PREFIX):
        return "oblique"
    return "unknown"


# =============================================================================
# INFERENCE
# =============================================================================

def predict_with_thresholds(model, image_path, classes, base_threshold=0.25):
    """
    Inference YOLO avec seuils de confiance par classe.

    On utilise un seuil bas global (0.25) pour ne pas manquer de detections,
    puis on filtre par le seuil specifique a chaque classe.
    """
    start = time.time()
    results = model.predict(image_path, conf=base_threshold, verbose=False)
    inference_time = time.time() - start

    result = results[0]
    preds = {
        "boxes":          [],
        "labels":         [],
        "scores":         [],
        "class_names":    [],
        "inference_time": inference_time,
    }

    if len(result.boxes) == 0:
        return preds

    boxes  = result.boxes.xyxy.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if label >= len(classes):
            continue
        class_name = classes[label]
        threshold  = CLASS_THRESHOLDS.get(class_name, 0.5)
        if score >= threshold:
            preds["boxes"].append(box)
            preds["labels"].append(label)
            preds["scores"].append(float(score))
            preds["class_names"].append(class_name)

    if preds["boxes"]:
        preds["boxes"]  = np.array(preds["boxes"])
        preds["scores"] = np.array(preds["scores"])

    return preds


# =============================================================================
# NMS FUSION (mode both)
# =============================================================================

def _nms(boxes, scores, iou_threshold=0.5):
    """NMS glouton sur un ensemble de boites."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds  = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def apply_nms_fusion(preds_list, iou_threshold=0.5):
    """
    Fusionner les predictions de plusieurs modeles via NMS.
    Utilise en mode 'both'.
    """
    total_time = sum(p.get("inference_time", 0) for p in preds_list)

    all_boxes   = []
    all_scores  = []
    all_labels  = []
    all_classes = []

    for preds in preds_list:
        boxes = preds.get("boxes", [])
        if len(boxes) == 0:
            continue
        if not isinstance(boxes, np.ndarray):
            boxes = np.array(boxes)
        scores = preds.get("scores", [])
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)
        for box, score, label, cls in zip(
            boxes, scores, preds.get("labels", []), preds.get("class_names", [])
        ):
            all_boxes.append(box)
            all_scores.append(float(score))
            all_labels.append(label)
            all_classes.append(cls)

    empty = {
        "boxes": [], "labels": [], "scores": [], "class_names": [],
        "inference_time": total_time,
    }

    if not all_boxes:
        return empty

    all_boxes  = np.array(all_boxes)
    all_scores = np.array(all_scores)

    keep = _nms(all_boxes, all_scores, iou_threshold)

    return {
        "boxes":          all_boxes[keep],
        "labels":         [all_labels[i]  for i in keep],
        "scores":         all_scores[keep],
        "class_names":    [all_classes[i] for i in keep],
        "inference_time": total_time,
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def visualize(image, preds, output_path=None, show=True, title_suffix=""):
    """Dessiner les boites et sauvegarder / afficher."""

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(image)

    boxes       = preds.get("boxes", [])
    class_names = preds.get("class_names", [])
    scores      = preds.get("scores", [])

    if len(boxes) > 0:
        for box, cls, score in zip(boxes, class_names, scores):
            color = [c / 255 for c in COLORS.get(cls, (128, 128, 128))]
            x1, y1, x2, y2 = box
            axes[1].add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor=color, facecolor="none",
                )
            )
            axes[1].text(
                x1, max(0, y1 - 5),
                f"{cls}\n{score:.2f}",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.8),
            )

    t     = preds.get("inference_time", 0)
    t_str = f"{t * 1000:.1f} ms" if t < 1 else f"{t:.2f} s"
    axes[1].set_title(f"YOLO Dual : {len(boxes)} objet(s) | {t_str}{title_suffix}")
    axes[1].axis("off")

    legend = [
        patches.Patch(facecolor=[c / 255 for c in col], label=name)
        for name, col in COLORS.items()
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# RAPPORT
# =============================================================================

def generate_report(preds, image_name, model_used):
    """Generer un rapport JSON pour une image."""

    report = {
        "image":             image_name,
        "model":             f"YOLO_dual_{model_used}",
        "timestamp":         datetime.now().isoformat(),
        "inference_time_ms": preds.get("inference_time", 0) * 1000,
        "total_objects":     len(preds.get("boxes", [])),
        "by_class":          {c: {"count": 0} for c in COLORS},
        "detections":        [],
    }

    boxes    = preds.get("boxes", [])
    classes  = preds.get("class_names", [])
    scores   = preds.get("scores", [])

    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        if cls in report["by_class"]:
            report["by_class"][cls]["count"] += 1
        box_list = box.tolist() if hasattr(box, "tolist") else list(box)
        report["detections"].append({
            "id":         i,
            "class":      cls,
            "confidence": float(score),
            "bbox":       box_list,
        })

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inference YOLO duale — routing nadir/oblique",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nadir-model",   default=None, help="Chemin modele nadir (.pt)")
    parser.add_argument("--oblique-model", default=None, help="Chemin modele oblique (.pt)")
    parser.add_argument(
        "--input",
        default=os.getenv("DETECTION_TEST_IMAGES_DIR", "../test"),
        help="Image unique ou dossier d'images",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("PREDICTIONS_DIR", "./predictions"),
        help="Dossier de sortie",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "nadir", "oblique", "both"],
        default="auto",
        help=(
            "auto    : routing par nom de fichier (Production_* / Snapshot_*)\n"
            "nadir   : forcer modele nadir\n"
            "oblique : forcer modele oblique\n"
            "both    : appliquer les deux modeles avec fusion NMS"
        ),
    )
    parser.add_argument("--no-display", action="store_true", help="Ne pas afficher les images")
    args = parser.parse_args()

    output_base = os.getenv("OUTPUT_DIR", "./output")

    print("=" * 65)
    print("   INFERENCE DUALE YOLO — Toitures Cadastrales")
    print("=" * 65)

    # -------------------------------------------------------------------
    # Charger les modeles
    # -------------------------------------------------------------------

    need_nadir   = args.mode in ("auto", "nadir",   "both")
    need_oblique = args.mode in ("auto", "oblique", "both")

    nadir_model     = None
    oblique_model   = None
    nadir_classes   = []
    oblique_classes = []

    if need_nadir:
        path = args.nadir_model or find_model("nadir", output_base)
        if path and os.path.exists(path):
            nadir_model   = YOLO(path)
            nadir_classes = load_classes_from_yaml("classes_nadir.yaml")
            if not nadir_classes:
                nadir_classes = ["panneau_solaire"]
            print(f"\nModele nadir   : {path}")
            print(f"Classes        : {nadir_classes}")
        else:
            print(f"\nAvertissement — modele nadir non trouve (cherche : {path})")
            if args.mode == "nadir":
                print("Erreur : mode 'nadir' requiert le modele nadir.")
                return

    if need_oblique:
        path = args.oblique_model or find_model("oblique", output_base)
        if path and os.path.exists(path):
            oblique_model   = YOLO(path)
            oblique_classes = load_classes_from_yaml("classes_oblique.yaml")
            if not oblique_classes:
                oblique_classes = ["batiment_peint", "batiment_non_enduit", "batiment_enduit"]
            print(f"\nModele oblique : {path}")
            print(f"Classes        : {oblique_classes}")
        else:
            print(f"\nAvertissement — modele oblique non trouve (cherche : {path})")
            if args.mode == "oblique":
                print("Erreur : mode 'oblique' requiert le modele oblique.")
                return

    if nadir_model is None and oblique_model is None:
        print("\nErreur : aucun modele disponible. Lancez d'abord train.py.")
        return

    # -------------------------------------------------------------------
    # Collecter les images
    # -------------------------------------------------------------------

    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTS)
    else:
        images = [input_path]

    print(f"\n{len(images)} image(s) — mode : {args.mode}\n")
    os.makedirs(args.output, exist_ok=True)

    reports     = []
    start_total = time.time()

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}")

        image    = Image.open(str(img_path)).convert("RGB")
        img_type = detect_image_type(img_path.name)

        # Determiner quel(s) modele(s) utiliser
        if args.mode == "auto":
            if img_type == "nadir" and nadir_model:
                use_mode = "nadir"
            elif img_type == "oblique" and oblique_model:
                use_mode = "oblique"
            elif nadir_model:
                use_mode = "nadir"
                print(f"   Type inconnu — utilisation nadir par defaut")
            else:
                use_mode = "oblique"
                print(f"   Type inconnu — utilisation oblique par defaut")
        else:
            use_mode = args.mode

        # Inference
        if use_mode == "nadir" and nadir_model:
            preds      = predict_with_thresholds(nadir_model, str(img_path), nadir_classes)
            model_used = "nadir"

        elif use_mode == "oblique" and oblique_model:
            preds      = predict_with_thresholds(oblique_model, str(img_path), oblique_classes)
            model_used = "oblique"

        elif use_mode == "both":
            preds_list = []
            if nadir_model:
                preds_list.append(
                    predict_with_thresholds(nadir_model, str(img_path), nadir_classes)
                )
            if oblique_model:
                preds_list.append(
                    predict_with_thresholds(oblique_model, str(img_path), oblique_classes)
                )
            preds      = apply_nms_fusion(preds_list, iou_threshold=0.5)
            model_used = "both"

        else:
            # Fallback si le modele demande n'est pas disponible
            fb_model   = nadir_model or oblique_model
            fb_classes = nadir_classes or oblique_classes
            preds      = predict_with_thresholds(fb_model, str(img_path), fb_classes)
            model_used = "fallback"
            print(f"   Fallback vers le seul modele disponible")

        # Visualisation
        out_img   = os.path.join(args.output, f"{img_path.stem}_dual.png")
        title_sfx = f" [{img_type} -> {model_used}]"
        visualize(image, preds, out_img, show=not args.no_display, title_suffix=title_sfx)

        # Rapport
        report = generate_report(preds, img_path.name, model_used)
        reports.append(report)

        n_obj = len(preds.get("boxes", []))
        t_ms  = preds.get("inference_time", 0) * 1000
        print(f"   {n_obj} objet(s) | {t_ms:.1f} ms | modele : {model_used}")

    # -------------------------------------------------------------------
    # Sauvegarder les rapports
    # -------------------------------------------------------------------

    with open(os.path.join(args.output, "reports_dual.json"), "w") as f:
        json.dump(reports, f, indent=2)

    total_time = time.time() - start_total

    if len(images) > 1:
        summary = {
            "model":            "YOLO_dual",
            "mode":             args.mode,
            "timestamp":        datetime.now().isoformat(),
            "total_images":     len(reports),
            "total_time_s":     total_time,
            "avg_inference_ms": (
                sum(r["inference_time_ms"] for r in reports) / len(reports)
            ),
            "total_objects":    sum(r["total_objects"] for r in reports),
            "by_class":         {
                c: sum(r["by_class"].get(c, {}).get("count", 0) for r in reports)
                for c in COLORS
            },
        }
        with open(os.path.join(args.output, "summary_dual.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nResume :")
        print(f"   Images   : {summary['total_images']}")
        print(f"   Objets   : {summary['total_objects']}")
        print(f"   Temps    : {summary['avg_inference_ms']:.1f} ms/image")
        for cls, count in summary["by_class"].items():
            if count > 0:
                print(f"   {cls}: {count}")

    print(f"\nResultats : {args.output}")


if __name__ == "__main__":
    main()
