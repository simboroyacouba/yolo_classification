"""
Optimisation des hyperparametres YOLO avec Optuna.

Equivalent de KerasTuner pour Ultralytics YOLO (PyTorch).
Chaque essai entraine le modele pendant --tune-epochs epochs et optimise
le mAP@50 sur le split de validation.

Algorithme : TPE (Tree-structured Parzen Estimator) + MedianPruner.
Les etudes sont persistees dans une base SQLite (resumable).

Usage :
  python tune.py --mode oblique --n-trials 20
  python tune.py --mode nadir   --n-trials 20 --attention cbam
  python tune.py --mode oblique --n-trials 30 --tune-epochs 15
  python tune.py --mode oblique --resume          # reprendre une etude existante
"""

import os
import gc
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import optuna
from optuna.samplers import TPESampler
from optuna.pruners  import MedianPruner
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from train import (
    MODE_CLASSES,
    OVERSAMPLE_WEIGHTS_OBLIQUE,
    load_classes,
    prepare_yolo_dataset,
    make_staged_training_callback,
    _register_cbam,
    build_cbam_yaml,
    _load_pretrained_partial,
)


# =============================================================================
# ESPACE DE RECHERCHE
# =============================================================================

def _suggest_hparams(trial):
    """Definit l'espace de recherche des hyperparametres."""
    return {
        "lr0":           trial.suggest_float("lr0",          1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "momentum":      trial.suggest_float("momentum",     0.85, 0.98),
        "hsv_s":         trial.suggest_float("hsv_s",        0.1,  0.6),
        "hsv_v":         trial.suggest_float("hsv_v",        0.1,  0.5),
        "fliplr":        trial.suggest_float("fliplr",       0.0,  0.7),
        "freeze_epochs": trial.suggest_categorical("freeze_epochs", [0, 3, 5, 10]),
    }


# =============================================================================
# OBJECTIF OPTUNA
# =============================================================================

def make_objective(base_config, yaml_path):
    """
    Retourne la fonction objectif Optuna.
    Le dataset est prepare une seule fois (yaml_path partage entre trials).
    """

    def objective(trial):
        hp = _suggest_hparams(trial)

        trial_dir  = os.path.join(base_config["tune_dir"], f"trial_{trial.number:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        use_cbam   = base_config.get("use_attention", "none") == "cbam"
        model_name = f"{base_config['model_version']}{base_config['model_size']}.pt"

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Charger le modele ---
        if use_cbam:
            _register_cbam()
            cbam_yaml = build_cbam_yaml(
                base_config["model_version"], base_config["model_size"], trial_dir
            )
            model = YOLO(cbam_yaml) if cbam_yaml else YOLO(model_name)
            if cbam_yaml and os.path.exists(model_name):
                _load_pretrained_partial(model, model_name)
        else:
            model = YOLO(model_name)

        # --- Callback de pruning ---
        best_map50 = [0.0]

        def pruning_cb(trainer):
            box = getattr(getattr(getattr(trainer, "validator", None), "metrics", None), "box", None)
            if box is None:
                return
            map50 = float(getattr(box, "map50", 0.0) or 0.0)
            best_map50[0] = max(best_map50[0], map50)
            trial.report(map50, trainer.epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model.add_callback("on_fit_epoch_end", pruning_cb)

        freeze_n = hp["freeze_epochs"]
        if freeze_n > 0:
            model.add_callback(
                "on_train_epoch_start",
                make_staged_training_callback(freeze_n),
            )

        # --- Entrainement ---
        try:
            results = model.train(
                data=yaml_path,
                epochs=base_config["tune_epochs"],
                batch=base_config["batch_size"],
                imgsz=base_config["image_size"],
                lr0=hp["lr0"],
                momentum=hp["momentum"],
                weight_decay=hp["weight_decay"],
                cos_lr=True,
                fliplr=hp["fliplr"],
                flipud=0.5,
                degrees=0.0,
                hsv_h=0.05,
                hsv_s=hp["hsv_s"],
                hsv_v=hp["hsv_v"],
                mosaic=0.0,
                mixup=0.0,
                freeze=10 if freeze_n > 0 else 0,
                project=trial_dir,
                name="run",
                seed=trial.number,
                verbose=False,
                save=False,
                plots=False,
                cache=False,
                workers=0,
            )
            map50 = float(results.results_dict.get("metrics/mAP50(B)", 0.0) or 0.0)

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"\n   [Trial {trial.number}] Erreur : {e}")
            map50 = 0.0
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                del model
            except Exception:
                pass

        return map50

    return objective


# =============================================================================
# RAPPORT FINAL
# =============================================================================

def _print_and_save_report(study, base_config):
    """Affiche et sauvegarde le rapport de l'etude."""
    best = study.best_trial

    print("\n" + "=" * 70)
    print("   RESULTATS DE L'OPTIMISATION")
    print("=" * 70)
    print(f"   Nombre d'essais termines : {len(study.trials)}")
    print(f"   Meilleur mAP@50          : {best.value:.4f}  (trial #{best.number})")
    print(f"\n   Meilleurs hyperparametres :")
    for k, v in best.params.items():
        print(f"      {k:<20} {v}")

    # Top 5 trials
    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value, reverse=True)
    print(f"\n   Top-5 trials :")
    print(f"   {'Trial':>6}  {'mAP@50':>8}  {'lr0':>10}  {'wd':>10}  {'momentum':>8}")
    print(f"   {'-'*55}")
    for t in completed[:5]:
        p = t.params
        print(
            f"   {t.number:>6}  {t.value:>8.4f}  "
            f"{p.get('lr0', 0):>10.5f}  {p.get('weight_decay', 0):>10.6f}  "
            f"{p.get('momentum', 0):>8.4f}"
        )

    # Sauvegarder les meilleurs hparams
    report = {
        "study_name":    study.study_name,
        "n_trials":      len(study.trials),
        "best_trial":    best.number,
        "best_map50":    best.value,
        "best_params":   best.params,
        "top5": [
            {"trial": t.number, "map50": t.value, "params": t.params}
            for t in completed[:5]
        ],
        "optimized_at":  datetime.now().isoformat(),
    }

    report_path = os.path.join(base_config["tune_dir"], "best_hparams.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n   Meilleurs hparams sauvegardes : {report_path}")
    print(f"\n   Lancer l'entrainement final :")
    p = best.params
    print(
        f"   python train.py --mode {base_config['mode']}"
        + (f" --attention {base_config['use_attention']}" if base_config.get("use_attention") != "none" else "")
        + f"  # puis editer .env : LR={p.get('lr0', ''):.5f}"
    )
    print("=" * 70)

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Optimisation hyperparametres YOLO — Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["nadir", "oblique", "all"],
        default="oblique",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Nombre d'essais Optuna.",
    )
    parser.add_argument(
        "--tune-epochs",
        type=int,
        default=10,
        help="Epochs par essai (moins que l'entrainement final).",
    )
    parser.add_argument(
        "--attention",
        choices=["none", "cbam"],
        default="none",
        help="Mecanisme d'attention.",
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Nom de l'etude Optuna (default : yolo_<mode>_<date>).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reprendre une etude existante (meme --study-name).",
    )
    parser.add_argument("--images-dir",       default=None)
    parser.add_argument("--annotations-file", default=None)
    parser.add_argument("--classes-file",     default=None)
    parser.add_argument("--output-dir",       default=None)
    args = parser.parse_args()

    # ----- Config de base -----
    mode       = args.mode
    output_dir = args.output_dir or os.path.join(os.getenv("OUTPUT_DIR", "./output"), mode)
    tune_dir   = os.path.join(output_dir, "tuning")
    os.makedirs(tune_dir, exist_ok=True)

    base_annotations = os.getenv(
        "DETECTION_DATASET_ANNOTATIONS_FILE",
        "../dataset1/annotations/instances_default.json",
    )
    ann_dir = os.path.dirname(os.path.abspath(base_annotations))

    if args.annotations_file:
        annotations_file = args.annotations_file
    elif mode == "nadir":
        annotations_file = os.path.join(ann_dir, "instances_nadir.json")
    elif mode == "oblique":
        annotations_file = os.path.join(ann_dir, "instances_oblique.json")
    else:
        annotations_file = base_annotations

    if args.classes_file:
        classes_file = args.classes_file
    elif mode == "nadir":
        classes_file = "classes_nadir.yaml"
    elif mode == "oblique":
        classes_file = "classes_oblique.yaml"
    else:
        classes_file = os.getenv("CLASSES_FILE", "classes.yaml")

    classes = load_classes(classes_file, mode=mode)
    if not classes:
        print(f"Erreur : aucune classe pour le mode '{mode}'")
        return

    base_config = {
        "mode":          mode,
        "images_dir":    args.images_dir or os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
        "annotations_file": annotations_file,
        "output_dir":    output_dir,
        "tune_dir":      tune_dir,
        "model_version": os.getenv("YOLO_VERSION", "yolo26"),
        "model_size":    os.getenv("YOLO_SIZE", "n"),
        "tune_epochs":   args.tune_epochs,
        "batch_size":    int(os.getenv("BATCH_SIZE", "2")),
        "image_size":    int(os.getenv("IMAGE_SIZE", "640")),
        "train_split":   float(os.getenv("TRAIN_SPLIT", "0.70")),
        "val_split":     float(os.getenv("VAL_SPLIT", "0.20")),
        "test_split":    float(os.getenv("TEST_SPLIT", "0.10")),
        "use_attention": args.attention,
        "classes":       classes,
    }

    print("=" * 70)
    print(f"   OPTUNA — Optimisation hyperparametres YOLO")
    print("=" * 70)
    print(f"   Mode         : {mode}")
    print(f"   Modele       : {base_config['model_version']}{base_config['model_size']}")
    print(f"   Attention    : {args.attention.upper()}")
    print(f"   Essais       : {args.n_trials}  x  {args.tune_epochs} epochs")
    print(f"   Dossier      : {tune_dir}")

    # ----- Preparer le dataset une seule fois -----
    print("\n📂 Preparation du dataset (une seule fois pour tous les essais)...")
    dataset_dir = os.path.join(tune_dir, "dataset")
    yaml_path, stats = prepare_yolo_dataset(
        base_config["images_dir"],
        base_config["annotations_file"],
        dataset_dir,
        classes,
        base_config["train_split"],
        base_config["val_split"],
        base_config["test_split"],
        mode=mode,
    )

    # ----- Creer ou reprendre l'etude -----
    study_name   = args.study_name or f"yolo_{mode}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    storage_path = os.path.join(tune_dir, "optuna_study.db")
    storage_url  = f"sqlite:///{storage_path}"

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=args.resume,
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=5),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1),
    )

    already_done = len([t for t in study.trials if t.value is not None])
    n_remaining  = max(0, args.n_trials - already_done)

    if n_remaining == 0:
        print(f"\n   Etude deja complete ({already_done} essais). Utilisez --resume pour continuer.")
    else:
        if already_done > 0:
            print(f"\n   Reprise : {already_done} essais deja faits, {n_remaining} restants.")

        print(f"\n🔍 Lancement de l'optimisation ({n_remaining} essais)...\n")
        study.optimize(
            make_objective(base_config, yaml_path),
            n_trials=n_remaining,
            show_progress_bar=True,
            catch=(Exception,),
        )

    _print_and_save_report(study, base_config)


if __name__ == "__main__":
    main()
