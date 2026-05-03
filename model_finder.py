"""
Decouverte automatique des modeles et datasets YOLO par mode (nadir / oblique).

Priorite de recherche :
  1. model_info_{mode}.json  — ecrit par train.py, source de verite
  2. runs/detect/{mode}/     — nouvelle convention (project=runs/detect/{mode})
  3. output/{mode}/          — copie best_model.pt faite par train.py
  4. Erreur explicite         — pas de fallback silencieux vers un mauvais modele
"""

import os
import json
from pathlib import Path


def find_model_and_data(mode, output_base=None, script_dir=None):
    """
    Retourne (model_path, dataset_yaml) pour le mode donne.
    Leve une erreur descriptive si le modele n'est pas trouve.
    """
    if script_dir is None:
        script_dir = Path(__file__).parent
    else:
        script_dir = Path(script_dir)

    if output_base is None:
        output_base = str(script_dir / "output")

    model_path = None
    data_yaml  = None

    # ------------------------------------------------------------------
    # 1. model_info_{mode}.json genere par train.py
    # ------------------------------------------------------------------
    info_path = Path(output_base) / f"model_info_{mode}.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        candidate = info.get("best_model")
        yaml_cand = info.get("dataset_yaml")
        if candidate and Path(candidate).exists():
            model_path = candidate
            print(f"   [{mode}] model_info.json -> {model_path}")
        if yaml_cand and Path(yaml_cand).exists():
            data_yaml = yaml_cand

    # ------------------------------------------------------------------
    # 2. runs/detect/{mode}/train* (nouvelle convention project/name)
    # ------------------------------------------------------------------
    if not model_path:
        runs_mode = script_dir / "runs" / "detect" / mode
        if runs_mode.exists():
            candidates = sorted(
                [f for f in runs_mode.iterdir() if f.is_dir()],
                key=lambda x: x.stat().st_mtime,
            )
            for folder in reversed(candidates):
                pt = folder / "weights" / "best.pt"
                if pt.exists():
                    model_path = str(pt)
                    print(f"   [{mode}] runs/detect/{mode}/ -> {model_path}")
                    break

    # ------------------------------------------------------------------
    # 3. output/{mode}/best_model.pt copie par train.py
    # ------------------------------------------------------------------
    if not model_path:
        candidate = Path(output_base) / mode / "best_model.pt"
        if candidate.exists():
            model_path = str(candidate)
            print(f"   [{mode}] output/{mode}/ -> {model_path}")

    # ------------------------------------------------------------------
    # Dataset yaml — output/{mode}/dataset/dataset.yaml
    # ------------------------------------------------------------------
    if not data_yaml:
        yaml_candidate = Path(output_base) / mode / "dataset" / "dataset.yaml"
        if yaml_candidate.exists():
            data_yaml = str(yaml_candidate)

    # Chercher dans runs/detect/{mode}/train*/
    if not data_yaml and model_path:
        train_dir = Path(model_path).parent.parent
        yaml_candidate = train_dir / "dataset" / "dataset.yaml"
        if yaml_candidate.exists():
            data_yaml = str(yaml_candidate)

    # ------------------------------------------------------------------
    # Erreur explicite — pas de fallback vers un mauvais modele
    # ------------------------------------------------------------------
    if not model_path:
        raise FileNotFoundError(
            f"\n[ERREUR] Modele '{mode}' introuvable.\n"
            f"Lancez d'abord : python train.py --mode {mode}\n"
            f"Ou passez le chemin manuellement : --{mode}-model chemin/vers/best.pt"
        )

    if not data_yaml:
        raise FileNotFoundError(
            f"\n[ERREUR] dataset.yaml pour '{mode}' introuvable.\n"
            f"Relancez : python train.py --mode {mode}\n"
            f"Ou passez le chemin manuellement : --{mode}-data chemin/vers/dataset.yaml"
        )

    return model_path, data_yaml
