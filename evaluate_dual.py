"""
Evaluation duale YOLO — nadir et oblique separement puis resultats combines

Evaluation :
  - Modele nadir   : evalue sur le test set nadir   (panneau_solaire)
  - Modele oblique : evalue sur le test set oblique (batiments)
  - Rapport combine : AP@50 par classe + metriques globales

Usage :
  python evaluate_dual.py
  python evaluate_dual.py --nadir-model  output/nadir/.../best_model.pt \\
                           --oblique-model output/oblique/.../best_model.pt
  python evaluate_dual.py --skip-oblique
"""

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
# CONSTANTES
# =============================================================================

NADIR_CLASSES   = ["panneau_solaire"]
OBLIQUE_CLASSES = ["batiment_peint", "batiment_non_enduit", "batiment_enduit"]
ALL_CLASSES     = NADIR_CLASSES + OBLIQUE_CLASSES

COLORS_MAP = {
    "nadir":   "#2196F3",   # Bleu
    "oblique": "#FF9800",   # Orange
}


# =============================================================================
# DECOUVERTE AUTOMATIQUE
# =============================================================================

def find_model_and_data(mode, output_base="./output"):
    """Trouver le chemin du modele et du dataset.yaml pour un mode donne."""

    model_path = None
    data_yaml  = None

    # 1. model_info_{mode}.json genere par train.py
    info_path = os.path.join(output_base, f"model_info_{mode}.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        candidate  = info.get("best_model")
        data_yaml  = info.get("dataset_yaml")
        if candidate and os.path.exists(candidate):
            model_path = candidate
            print(f"   [{mode}] model_info.json -> {model_path}")

    # 2. Parcourir output/{mode}/
    if not model_path:
        base_dir = os.path.join(output_base, mode)
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                if "weights" in dirs:
                    pt = os.path.join(root, "weights", "best.pt")
                    if os.path.exists(pt):
                        model_path = pt
                        print(f"   [{mode}] scan -> {model_path}")
                        break

    # 3. dataset.yaml
    if not data_yaml:
        yaml_candidate = os.path.join(output_base, mode, "dataset", "dataset.yaml")
        if os.path.exists(yaml_candidate):
            data_yaml = yaml_candidate

    return model_path, data_yaml


# =============================================================================
# EVALUATION D'UN MODELE
# =============================================================================

def evaluate_model(model_path, data_yaml, mode, output_dir):
    """
    Evaluer un modele YOLO sur son test set.
    Retourne un dict de metriques ou None si echec.
    """

    print(f"\n{'='*65}")
    print(f"   Evaluation : {mode.upper()}")
    print(f"{'='*65}")

    if not model_path or not os.path.exists(model_path):
        print(f"   Modele non trouve : {model_path}")
        print(f"   Conseil : lancez  python train.py --mode {mode}")
        return None

    if not data_yaml or not os.path.exists(data_yaml):
        print(f"   dataset.yaml non trouve : {data_yaml}")
        print(f"   Conseil : lancez  python train.py --mode {mode}")
        return None

    print(f"   Modele  : {model_path}")
    print(f"   Dataset : {data_yaml}")

    model = YOLO(model_path)

    # Evaluation sur le test set
    print("\n   Evaluation sur le test set...")
    results = model.val(data=data_yaml, split="test", verbose=False)

    box   = results.box
    names = results.names

    # Metriques par classe
    per_class_ap50 = {}
    ap_class_index = getattr(box, "ap_class_index", None)
    ap_matrix      = getattr(box, "ap",             None)

    if ap_matrix is not None and ap_class_index is not None:
        ap50 = ap_matrix[:, 0] if ap_matrix.ndim == 2 else ap_matrix
        for idx, ap in zip(ap_class_index, ap50):
            cls_name = names.get(int(idx), f"class_{idx}")
            per_class_ap50[cls_name] = float(ap)

    metrics = {
        "mode":             mode,
        "model":            model_path,
        "dataset_yaml":     data_yaml,
        "mAP50":            float(box.map50),
        "mAP50_95":         float(box.map),
        "precision":        float(box.mp),
        "recall":           float(box.mr),
        "per_class_AP50":   per_class_ap50,
        "evaluated_at":     datetime.now().isoformat(),
    }

    # Afficher le tableau
    print(f"\n   {'Metrique':<28} {'Valeur':>8}")
    print(f"   {'-'*38}")
    print(f"   {'mAP@50':<28} {metrics['mAP50']:>8.4f}")
    print(f"   {'mAP@50:95':<28} {metrics['mAP50_95']:>8.4f}")
    print(f"   {'Precision':<28} {metrics['precision']:>8.4f}")
    print(f"   {'Recall':<28} {metrics['recall']:>8.4f}")

    if per_class_ap50:
        print(f"\n   AP@50 par classe :")
        for cls, ap in sorted(per_class_ap50.items()):
            bar    = "█" * int(ap * 20)
            status = "OK" if ap >= 0.5 else ("~" if ap >= 0.3 else "!!")
            print(f"      {cls:<25} {ap:.4f}  {bar}  [{status}]")

    # Sauvegarder
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"metrics_{mode}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n   Metriques sauvegardees : {metrics_path}")

    return metrics


# =============================================================================
# RAPPORT COMBINE
# =============================================================================

def print_combined_report(nadir_metrics, oblique_metrics, output_dir):
    """Afficher et sauvegarder le rapport combine."""

    print("\n" + "=" * 65)
    print("   RAPPORT COMBINE — Evaluation Duale YOLO")
    print("=" * 65)

    # Tableau par classe
    print(f"\n   {'Classe':<25} {'Modele':<10} {'AP@50':>8}  Statut")
    print(f"   {'-'*55}")

    all_results = {}

    if nadir_metrics:
        for cls, ap in nadir_metrics.get("per_class_AP50", {}).items():
            all_results[cls] = {"ap": ap, "model": "nadir"}

    if oblique_metrics:
        for cls, ap in oblique_metrics.get("per_class_AP50", {}).items():
            all_results[cls] = {"ap": ap, "model": "oblique"}

    for cls in ALL_CLASSES:
        if cls in all_results:
            ap    = all_results[cls]["ap"]
            model = all_results[cls]["model"]
            if ap >= 0.5:
                status = "[OK]"
            elif ap >= 0.3:
                status = "[~]"
            else:
                status = "[!!]"
            print(f"   {cls:<25} {model:<10} {ap:>8.4f}  {status}")
        else:
            print(f"   {cls:<25} {'N/A':<10} {'—':>8}")

    print(f"   {'-'*55}")

    # Metriques globales par modele
    for label, m in [("NADIR", nadir_metrics), ("OBLIQUE", oblique_metrics)]:
        if m:
            print(f"\n   [{label}]")
            print(f"   mAP@50    : {m['mAP50']:.4f} ({m['mAP50'] * 100:.2f}%)")
            print(f"   mAP@50:95 : {m['mAP50_95']:.4f}")
            print(f"   Precision : {m['precision']:.4f}")
            print(f"   Recall    : {m['recall']:.4f}")

    # Moyenne globale
    all_maps = []
    if nadir_metrics:
        all_maps.append(nadir_metrics["mAP50"])
    if oblique_metrics:
        all_maps.append(oblique_metrics["mAP50"])

    if all_maps:
        mean_map = float(np.mean(all_maps))
        print(f"\n   mAP@50 moyen (dual) : {mean_map:.4f} ({mean_map * 100:.2f}%)")
    else:
        mean_map = None

    print("=" * 65)

    # Sauvegarder le rapport combine
    combined = {
        "timestamp":      datetime.now().isoformat(),
        "nadir":          nadir_metrics,
        "oblique":        oblique_metrics,
        "combined_mAP50": mean_map,
        "per_class":      {
            cls: all_results[cls] for cls in ALL_CLASSES if cls in all_results
        },
    }

    report_path = os.path.join(output_dir, "metrics_combined.json")
    with open(report_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n   Rapport combine : {report_path}")

    return combined


# =============================================================================
# GRAPHIQUE
# =============================================================================

def plot_combined_metrics(nadir_metrics, oblique_metrics, output_dir):
    """Tracer l'AP@50 par classe pour les deux modeles."""

    entries = []

    for cls in ALL_CLASSES:
        for metrics, model in [(nadir_metrics, "nadir"), (oblique_metrics, "oblique")]:
            if metrics and cls in metrics.get("per_class_AP50", {}):
                entries.append({
                    "label": f"{cls}\n({model})",
                    "ap":    metrics["per_class_AP50"][cls],
                    "color": COLORS_MAP[model],
                })

    if not entries:
        return

    labels = [e["label"] for e in entries]
    ap50s  = [e["ap"]    for e in entries]
    colors = [e["color"] for e in entries]

    fig, ax = plt.subplots(figsize=(max(8, len(entries) * 1.5), 6))
    bars = ax.bar(range(len(labels)), ap50s, color=colors, edgecolor="white", linewidth=0.5)

    for bar, ap in zip(bars, ap50s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{ap:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("AP@50")
    ax.set_title("AP@50 par classe — Evaluation Duale YOLO")
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="red",    linestyle="--", alpha=0.5, label="Seuil 0.5")
    ax.axhline(y=0.3, color="orange", linestyle=":",  alpha=0.4, label="Seuil 0.3")
    ax.grid(True, alpha=0.3, axis="y")

    legend_handles = [
        mpatches.Patch(facecolor=COLORS_MAP[m], label=f"Modele {m}")
        for m in COLORS_MAP
    ] + [
        plt.Line2D([0], [0], color="red",    linestyle="--", label="Seuil 0.5"),
        plt.Line2D([0], [0], color="orange", linestyle=":",  label="Seuil 0.3"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "metrics_dual.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Graphique : {out_path}")


def plot_global_comparison(nadir_metrics, oblique_metrics, output_dir):
    """Graphique en barres des metriques globales (mAP50, Precision, Recall)."""

    available = {
        k: v for k, v in [("nadir", nadir_metrics), ("oblique", oblique_metrics)]
        if v is not None
    }

    if len(available) < 1:
        return

    metric_keys   = ["mAP50", "mAP50_95", "precision", "recall"]
    metric_labels = ["mAP@50", "mAP@50:95", "Precision", "Recall"]

    x     = np.arange(len(metric_keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, metrics) in enumerate(available.items()):
        vals = [metrics[k] for k in metric_keys]
        offset = (i - (len(available) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=f"Modele {name}",
                      color=COLORS_MAP[name], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.set_title("Metriques globales — Nadir vs Oblique")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "comparison_dual.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"   Comparaison : {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation duale YOLO — nadir + oblique",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--nadir-model",   default=None, help="Modele nadir (.pt)")
    parser.add_argument("--oblique-model", default=None, help="Modele oblique (.pt)")
    parser.add_argument("--nadir-data",    default=None, help="dataset.yaml nadir")
    parser.add_argument("--oblique-data",  default=None, help="dataset.yaml oblique")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("EVALUATION_DIR", "./evaluation"),
        help="Dossier de sortie",
    )
    parser.add_argument("--skip-nadir",   action="store_true", help="Ignorer evaluation nadir")
    parser.add_argument("--skip-oblique", action="store_true", help="Ignorer evaluation oblique")
    args = parser.parse_args()

    output_base = os.getenv("OUTPUT_DIR", "./output")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("   EVALUATION DUALE YOLO — Toitures Cadastrales")
    print("=" * 65)

    nadir_metrics   = None
    oblique_metrics = None

    # -------------------------------------------------------------------
    # Evaluation nadir
    # -------------------------------------------------------------------

    if not args.skip_nadir:
        nadir_model_path, nadir_yaml = find_model_and_data("nadir", output_base)

        if args.nadir_model:
            nadir_model_path = args.nadir_model
        if args.nadir_data:
            nadir_yaml = args.nadir_data

        nadir_metrics = evaluate_model(
            nadir_model_path, nadir_yaml, "nadir", args.output_dir
        )

    # -------------------------------------------------------------------
    # Evaluation oblique
    # -------------------------------------------------------------------

    if not args.skip_oblique:
        oblique_model_path, oblique_yaml = find_model_and_data("oblique", output_base)

        if args.oblique_model:
            oblique_model_path = args.oblique_model
        if args.oblique_data:
            oblique_yaml = args.oblique_data

        oblique_metrics = evaluate_model(
            oblique_model_path, oblique_yaml, "oblique", args.output_dir
        )

    # -------------------------------------------------------------------
    # Rapport combine et graphiques
    # -------------------------------------------------------------------

    if nadir_metrics or oblique_metrics:
        print_combined_report(nadir_metrics, oblique_metrics, args.output_dir)
        plot_combined_metrics(nadir_metrics, oblique_metrics, args.output_dir)
        plot_global_comparison(nadir_metrics, oblique_metrics, args.output_dir)
    else:
        print("\n   Aucune metrique disponible.")
        print("   Lancez d'abord :")
        print("      python train.py --mode nadir")
        print("      python train.py --mode oblique")


if __name__ == "__main__":
    main()
