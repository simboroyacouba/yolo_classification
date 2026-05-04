"""
Evaluation YOLO unifie — un seul modele, toutes classes

Evalue le modele entraine par train_unified.py sur son test set.
Produit : metriques JSON, graphique AP@50 par classe, rapport texte.

Usage :
  python evaluate_unified.py
  python evaluate_unified.py --model output/train_unified/best_model.pt
  python evaluate_unified.py --model best.pt --data output/dataset/dataset.yaml
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# DECOUVERTE AUTOMATIQUE DU MODELE
# =============================================================================

def find_unified_model(output_base=None, script_dir=None):
    """
    Retourne (model_path, dataset_yaml) pour le modele unifie.
    Priorite :
      1. model_info_unified.json ecrit par train_unified.py
      2. train_unified/ dans les dossiers de sortie
      3. Erreur explicite
    """
    if script_dir is None:
        script_dir = Path(__file__).parent
    else:
        script_dir = Path(script_dir)

    if output_base is None:
        output_base = str(script_dir / "output")

    model_path = None
    data_yaml  = None

    # 1. model_info_unified.json
    info_path = Path(output_base) / "model_info_unified.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        candidate = info.get("best_model")
        yaml_cand = info.get("dataset_yaml")
        if candidate and Path(candidate).exists():
            model_path = candidate
            print(f"   model_info_unified.json -> {model_path}")
        if yaml_cand and Path(yaml_cand).exists():
            data_yaml = yaml_cand

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

    # Chercher dataset.yaml dans un chemin contenant unified
    if not data_yaml:
        yaml_cand = Path(output_base) / "dataset" / "dataset.yaml"
        if yaml_cand.exists():
            data_yaml = str(yaml_cand)

    if not data_yaml:
        all_yamls = sorted(script_dir.rglob("dataset.yaml"), key=lambda p: p.stat().st_mtime)
        for y in reversed(all_yamls):
            if "unified" in str(y).lower():
                data_yaml = str(y)
                print(f"   dataset.yaml -> {data_yaml}")
                break

    if not data_yaml:
        raise FileNotFoundError(
            "\n[ERREUR] dataset.yaml introuvable.\n"
            "Relancez : python train_unified.py\n"
            "Ou passez le chemin manuellement : --data chemin/vers/dataset.yaml"
        )

    # Patcher le path du yaml si le chemin stocke n'existe plus
    import yaml as _yaml
    with open(data_yaml) as f:
        cfg = _yaml.safe_load(f)
    yaml_dir     = str(Path(data_yaml).parent)
    stored_path  = str(cfg.get("path", "."))
    if not Path(stored_path).exists() or stored_path == ".":
        cfg["path"] = yaml_dir
        with open(data_yaml, "w") as f:
            _yaml.dump(cfg, f, default_flow_style=False)
        print(f"   dataset.yaml patche : path -> {yaml_dir}")

    return model_path, data_yaml


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model_path, data_yaml, output_dir):
    """Evaluer le modele unifie sur le test set et retourner les metriques."""

    print(f"\n{'='*65}")
    print(f"   Evaluation — Modele Unifie")
    print(f"{'='*65}")
    print(f"   Modele  : {model_path}")
    print(f"   Dataset : {data_yaml}")

    model = YOLO(model_path)

    try:
        model_info = model.info(verbose=False)
        n_params_M = float(model_info[1]) / 1e6
    except Exception:
        n_params_M = None

    print("\n   Evaluation sur le test set...")
    results = model.val(data=data_yaml, split="test", verbose=False)

    box   = results.box
    names = results.names

    speed        = getattr(results, "speed", {}) or {}
    inference_ms = float(speed.get("inference", 0)) or None
    fps_gpu      = (1000.0 / inference_ms) if inference_ms else None

    ap_class_index = getattr(box, "ap_class_index", None)
    ap_matrix      = getattr(box, "ap",             None)
    p_per_cls      = getattr(box, "p",              None)
    r_per_cls      = getattr(box, "r",              None)

    per_class_ap50      = {}
    per_class_precision = {}
    per_class_recall    = {}
    per_class_f1        = {}

    if ap_matrix is not None and ap_class_index is not None:
        ap50 = ap_matrix[:, 0] if ap_matrix.ndim == 2 else ap_matrix
        for i, (idx, ap) in enumerate(zip(ap_class_index, ap50)):
            cls_name = names.get(int(idx), f"class_{idx}")
            per_class_ap50[cls_name] = float(ap)

            pc = float(p_per_cls[i]) if p_per_cls is not None and i < len(p_per_cls) else None
            rc = float(r_per_cls[i]) if r_per_cls is not None and i < len(r_per_cls) else None

            if pc is not None:
                per_class_precision[cls_name] = pc
            if rc is not None:
                per_class_recall[cls_name] = rc
            if pc is not None and rc is not None and (pc + rc) > 0:
                per_class_f1[cls_name] = 2 * pc * rc / (pc + rc)

    p, r = float(box.mp), float(box.mr)
    f1_score = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    metrics = {
        "model":                model_path,
        "dataset_yaml":         data_yaml,
        "mAP50":                float(box.map50),
        "mAP50_95":             float(box.map),
        "precision":            p,
        "recall":               r,
        "f1_score":             f1_score,
        "inference_ms":         inference_ms,
        "fps_gpu":              fps_gpu,
        "params_M":             n_params_M,
        "per_class_AP50":       per_class_ap50,
        "per_class_precision":  per_class_precision,
        "per_class_recall":     per_class_recall,
        "per_class_f1":         per_class_f1,
        "evaluated_at":         datetime.now().isoformat(),
    }

    # Affichage tableau global
    print(f"\n   {'Metrique':<32} {'Valeur':>10}")
    print(f"   {'-'*44}")
    print(f"   {'mAP@50':<32} {metrics['mAP50']:>10.4f}")
    print(f"   {'mAP@50:95':<32} {metrics['mAP50_95']:>10.4f}")
    print(f"   {'Precision':<32} {metrics['precision']:>10.4f}")
    print(f"   {'Recall':<32} {metrics['recall']:>10.4f}")
    print(f"   {'F1 Score':<32} {metrics['f1_score']:>10.4f}")
    if inference_ms is not None:
        print(f"   {'Vitesse Inference (ms)':<32} {inference_ms:>10.2f}")
    if fps_gpu is not None:
        print(f"   {'FPS GPU':<32} {fps_gpu:>10.1f}")
    if n_params_M is not None:
        print(f"   {'Parametres (M)':<32} {n_params_M:>10.2f}")

    # Affichage tableau par classe
    if per_class_ap50:
        print(f"\n   {'Classe':<25} {'AP@50':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}  Statut")
        print(f"   {'-'*62}")
        for cls in sorted(per_class_ap50.keys()):
            ap  = per_class_ap50.get(cls, float("nan"))
            pc  = per_class_precision.get(cls, float("nan"))
            rc  = per_class_recall.get(cls, float("nan"))
            f1c = per_class_f1.get(cls, float("nan"))
            status = "OK" if ap >= 0.5 else ("~" if ap >= 0.3 else "!!")
            print(f"      {cls:<25} {ap:>7.4f} {pc:>7.4f} {rc:>7.4f} {f1c:>7.4f}  [{status}]")

    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "metrics_unified.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   Metriques : {metrics_path}")

    return metrics


# =============================================================================
# GRAPHIQUES
# =============================================================================

def plot_per_class_ap(metrics, output_dir):
    """Graphique AP@50 par classe."""

    per_class = metrics.get("per_class_AP50", {})
    if not per_class:
        return

    classes = sorted(per_class.keys())
    ap50s   = [per_class[c] for c in classes]
    colors  = ["#4CAF50" if ap >= 0.5 else ("#FF9800" if ap >= 0.3 else "#F44336")
               for ap in ap50s]

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.6), 5))
    bars = ax.bar(classes, ap50s, color=colors, edgecolor="white", linewidth=0.5)

    for bar, ap in zip(bars, ap50s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{ap:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_ylabel("AP@50")
    ax.set_title("AP@50 par classe — Modele Unifie YOLO")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="red",    linestyle="--", alpha=0.5, label="Seuil 0.5")
    ax.axhline(y=0.3, color="orange", linestyle=":",  alpha=0.4, label="Seuil 0.3")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_unified.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Graphique : {out_path}")


def plot_global_metrics(metrics, output_dir):
    """Graphique en barres des metriques globales."""

    keys   = ["mAP50", "mAP50_95", "precision", "recall", "f1_score"]
    labels = ["mAP@50", "mAP@50:95", "Precision", "Recall", "F1"]
    vals   = [metrics.get(k, 0) for k in keys]
    colors = ["#2196F3", "#1565C0", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.5)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{v:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Score")
    ax.set_title("Metriques globales — Modele Unifie YOLO")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="Seuil 0.5")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_unified_global.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Metriques globales : {out_path}")


# =============================================================================
# RAPPORT TEXTE
# =============================================================================

def write_report(metrics, output_dir):
    """Ecrire un rapport texte resumant l'evaluation."""

    report_path = os.path.join(output_dir, "evaluation_report_unified.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("YOLO — Modele Unifie — Rapport d'Evaluation\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date          : {metrics['evaluated_at']}\n")
        f.write(f"Modele        : {metrics['model']}\n")
        f.write(f"Dataset       : {metrics['dataset_yaml']}\n\n")
        f.write(f"mAP@50        : {metrics['mAP50']:.4f}\n")
        f.write(f"mAP@50:95     : {metrics['mAP50_95']:.4f}\n")
        f.write(f"Precision     : {metrics['precision']:.4f}\n")
        f.write(f"Recall        : {metrics['recall']:.4f}\n")
        f.write(f"F1 Score      : {metrics['f1_score']:.4f}\n")
        if metrics.get("inference_ms"):
            f.write(f"Inference (ms): {metrics['inference_ms']:.2f}\n")
        if metrics.get("fps_gpu"):
            f.write(f"FPS GPU       : {metrics['fps_gpu']:.1f}\n")
        if metrics.get("params_M"):
            f.write(f"Parametres    : {metrics['params_M']:.2f} M\n")
        f.write("\nAP@50 par classe :\n")
        f.write("-" * 40 + "\n")
        for cls in sorted(metrics.get("per_class_AP50", {})):
            ap  = metrics["per_class_AP50"][cls]
            f1c = metrics.get("per_class_f1", {}).get(cls, float("nan"))
            f.write(f"  {cls:<28} AP@50={ap:.4f}  F1={f1c:.4f}\n")

    print(f"   Rapport texte : {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation YOLO unifie — un seul modele, toutes classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      default=None, help="Chemin vers le modele (.pt)")
    parser.add_argument("--data",       default=None, help="Chemin vers dataset.yaml")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("EVALUATION_DIR", "./evaluation_unified"),
        help="Dossier de sortie",
    )
    args = parser.parse_args()

    output_base = os.getenv("OUTPUT_DIR", "./output")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("   EVALUATION YOLO — Modele Unifie")
    print("=" * 65)

    # Trouver le modele et le dataset
    if args.model and args.data:
        model_path = args.model
        data_yaml  = args.data
    else:
        try:
            model_path, data_yaml = find_unified_model(output_base)
        except FileNotFoundError as e:
            print(e)
            return
        if args.model:
            model_path = args.model
        if args.data:
            data_yaml = args.data

    metrics = evaluate_model(model_path, data_yaml, args.output_dir)

    if metrics:
        plot_per_class_ap(metrics, args.output_dir)
        plot_global_metrics(metrics, args.output_dir)
        write_report(metrics, args.output_dir)

        print("\n" + "=" * 65)
        print("   RESUME")
        print("=" * 65)
        print(f"   mAP@50    : {metrics['mAP50']:.4f} ({metrics['mAP50'] * 100:.2f}%)")
        print(f"   mAP@50:95 : {metrics['mAP50_95']:.4f}")
        print(f"   Precision : {metrics['precision']:.4f}")
        print(f"   Recall    : {metrics['recall']:.4f}")
        print(f"   F1        : {metrics['f1_score']:.4f}")
        print(f"   Resultats : {args.output_dir}")
        print("=" * 65)


if __name__ == "__main__":
    main()
