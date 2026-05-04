"""
Inference YOLO unifie — un seul modele, toutes classes

Applique le modele entraine par train_unified.py sur une image ou un dossier.
Pas de routing nadir / oblique : toutes les images sont traitees par le meme modele.

Seuils de confiance par classe (ajustables dans CLASS_THRESHOLDS) :
  panneau_solaire     = 0.50
  batiment_peint      = 0.30
  batiment_non_enduit = 0.35
  batiment_enduit     = 0.35
  menuiserie_metallique = 0.40

Usage :
  python inference_unified.py --input ../test
  python inference_unified.py --input image.jpg --display
  python inference_unified.py --model output/train_unified/best_model.pt --input ../test
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

CLASS_THRESHOLDS = {
    "panneau_solaire":       0.50,
    "batiment_peint":        0.30,
    "batiment_non_enduit":   0.35,
    "batiment_enduit":       0.35,
    "menuiserie_metallique": 0.40,
}

COLORS = {
    "panneau_solaire":       (255, 215,   0),   # Or
    "batiment_peint":        (  0, 200,  83),   # Vert
    "batiment_non_enduit":   ( 33, 150, 243),   # Bleu
    "batiment_enduit":       (255, 152,   0),   # Orange
    "menuiserie_metallique": (156,  39, 176),   # Violet
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# =============================================================================
# DECOUVERTE AUTOMATIQUE DU MODELE
# =============================================================================

def find_unified_model(output_base=None, script_dir=None):
    """Trouver automatiquement le modele unifie."""

    if script_dir is None:
        script_dir = Path(__file__).parent
    else:
        script_dir = Path(script_dir)

    if output_base is None:
        output_base = str(script_dir / "output")

    model_path = None

    # 1. model_info_unified.json
    info_path = Path(output_base) / "model_info_unified.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        candidate = info.get("best_model")
        if candidate and Path(candidate).exists():
            model_path = candidate
            print(f"   model_info_unified.json -> {model_path}")

    # 2. Chercher best.pt dans un chemin contenant train_unified
    if not model_path:
        all_pts = sorted(script_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
        for pt in reversed(all_pts):
            if "unified" in str(pt).lower():
                model_path = str(pt)
                print(f"   Trouve -> {model_path}")
                break

    # 3. output/train_unified/best_model.pt
    if not model_path:
        candidate = Path(output_base) / "train_unified" / "best_model.pt"
        if candidate.exists():
            model_path = str(candidate)
            print(f"   output/train_unified/ -> {model_path}")

    if not model_path:
        raise FileNotFoundError(
            "\n[ERREUR] Modele unifie introuvable.\n"
            "Lancez d'abord : python train_unified.py\n"
            "Ou passez le chemin manuellement : --model chemin/vers/best_model.pt"
        )

    return model_path


def load_classes_from_model_info(output_base):
    """Charger les classes depuis model_info_unified.json."""
    info_path = Path(output_base) / "model_info_unified.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        return info.get("classes", [])
    return []


def load_classes_from_yaml(yaml_path):
    """Charger les classes depuis un fichier YAML."""
    if not yaml_path or not os.path.exists(yaml_path):
        return []
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [c for c in data.get("classes", []) if c != "__background__"]


# =============================================================================
# INFERENCE
# =============================================================================

def predict_with_thresholds(model, image_path, classes, base_threshold=0.20):
    """
    Inference YOLO avec seuils de confiance par classe.
    Utilise un seuil bas global puis filtre par classe.
    """
    start   = time.time()
    results = model.predict(image_path, conf=base_threshold, verbose=False)
    inference_time = time.time() - start

    result = results[0]
    preds  = {
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
        threshold  = CLASS_THRESHOLDS.get(class_name, 0.50)
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
# VISUALISATION
# =============================================================================

def visualize(image, preds, output_path=None, show=True, image_name=""):
    """Dessiner les boites de detection et sauvegarder / afficher."""

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
    n_obj = len(boxes) if hasattr(boxes, "__len__") else 0
    axes[1].set_title(f"YOLO Unifie : {n_obj} objet(s) | {t_str}")
    axes[1].axis("off")

    legend = [
        patches.Patch(facecolor=[c / 255 for c in col], label=name)
        for name, col in COLORS.items()
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=9)
    if image_name:
        fig.suptitle(image_name, fontsize=10, y=1.01)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# RAPPORT
# =============================================================================

def generate_report(preds, image_name):
    """Generer un rapport JSON pour une image."""

    report = {
        "image":             image_name,
        "model":             "YOLO_unified",
        "timestamp":         datetime.now().isoformat(),
        "inference_time_ms": preds.get("inference_time", 0) * 1000,
        "total_objects":     len(preds.get("boxes", [])) if hasattr(preds.get("boxes", []), "__len__") else 0,
        "by_class":          {c: {"count": 0} for c in COLORS},
        "detections":        [],
    }

    boxes   = preds.get("boxes", [])
    classes = preds.get("class_names", [])
    scores  = preds.get("scores", [])

    if len(boxes) == 0:
        return report

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
        description="Inference YOLO unifie — un seul modele, toutes classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",  default=None, help="Chemin modele (.pt)")
    parser.add_argument(
        "--input",
        default=os.getenv("DETECTION_TEST_IMAGES_DIR", "../test"),
        help="Image unique ou dossier d'images",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("PREDICTIONS_DIR", "./predictions_unified"),
        help="Dossier de sortie",
    )
    parser.add_argument(
        "--classes-file",
        default=None,
        help="Fichier classes YAML (facultatif, auto-detecte sinon)",
    )
    parser.add_argument("--display", action="store_true", help="Afficher les images")
    args = parser.parse_args()

    output_base = os.getenv("OUTPUT_DIR", "./output")

    print("=" * 65)
    print("   INFERENCE YOLO — Modele Unifie")
    print("=" * 65)

    # Trouver le modele
    model_path = args.model
    if not model_path:
        try:
            model_path = find_unified_model(output_base)
        except FileNotFoundError as e:
            print(e)
            return

    if not os.path.exists(model_path):
        print(f"Erreur : modele introuvable : {model_path}")
        return

    # Charger les classes
    classes = []
    if args.classes_file:
        classes = load_classes_from_yaml(args.classes_file)
    if not classes:
        classes = load_classes_from_model_info(output_base)
    if not classes:
        classes = load_classes_from_yaml(os.getenv("CLASSES_FILE", "classes.yaml"))
    if not classes:
        classes = [
            "panneau_solaire", "batiment_peint", "batiment_non_enduit",
            "batiment_enduit", "menuiserie_metallique",
        ]

    print(f"\nModele  : {model_path}")
    print(f"Classes : {classes}")

    model = YOLO(model_path)

    # Collecter les images
    input_path = Path(args.input)
    if input_path.is_dir():
        images = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMG_EXTS)
    else:
        images = [input_path]

    print(f"\n{len(images)} image(s)\n")
    os.makedirs(args.output, exist_ok=True)

    reports     = []
    start_total = time.time()

    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}")

        image = Image.open(str(img_path)).convert("RGB")
        preds = predict_with_thresholds(model, str(img_path), classes)

        out_img = os.path.join(args.output, f"{img_path.stem}_unified.png")
        visualize(image, preds, out_img, show=args.display, image_name=img_path.name)

        report = generate_report(preds, img_path.name)
        reports.append(report)

        n_obj = report["total_objects"]
        t_ms  = preds.get("inference_time", 0) * 1000
        print(f"   {n_obj} objet(s) | {t_ms:.1f} ms")
        if n_obj > 0:
            for cls, info in report["by_class"].items():
                if info["count"] > 0:
                    print(f"      {cls}: {info['count']}")

    # Sauvegarder les rapports
    reports_path = os.path.join(args.output, "reports_unified.json")
    with open(reports_path, "w") as f:
        json.dump(reports, f, indent=2)

    total_time = time.time() - start_total

    if len(images) > 1:
        summary = {
            "model":            "YOLO_unified",
            "timestamp":        datetime.now().isoformat(),
            "total_images":     len(reports),
            "total_time_s":     total_time,
            "avg_inference_ms": (
                sum(r["inference_time_ms"] for r in reports) / len(reports)
                if reports else 0
            ),
            "total_objects":    sum(r["total_objects"] for r in reports),
            "by_class":         {
                c: sum(r["by_class"].get(c, {}).get("count", 0) for r in reports)
                for c in COLORS
            },
        }
        summary_path = os.path.join(args.output, "summary_unified.json")
        with open(summary_path, "w") as f:
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
