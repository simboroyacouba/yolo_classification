"""
Split du dataset COCO en sous-datasets nadir et oblique.

  instances_nadir.json   : Production_*.png -> panneau_solaire UNIQUEMENT
  instances_oblique.json : Snapshot_*.jpg   -> batiment_peint, batiment_non_enduit, batiment_enduit
  classes_nadir.yaml     : fichier classes YOLO pour le modele nadir
  classes_oblique.yaml   : fichier classes YOLO pour le modele oblique

Usage:
  python split_dataset.py
  python split_dataset.py --annotations ../dataset1/annotations/instances_default.json
"""

import os
import json
import argparse
import yaml
from pathlib import Path

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

NADIR_CLASSES   = ["panneau_solaire"]
OBLIQUE_CLASSES = ["batiment_peint", "batiment_non_enduit", "batiment_enduit"]

COLORS = {
    "panneau_solaire":     [255, 0,   0],
    "batiment_peint":      [0,   255, 0],
    "batiment_non_enduit": [0,   0,   255],
    "batiment_enduit":     [255, 165, 0],
}

DESCRIPTIONS = {
    "panneau_solaire":     "Tole ondulee metallique (vue nadir)",
    "batiment_peint":      "Tole bac acier (vue oblique)",
    "batiment_non_enduit": "Tuiles ou beton non enduit (vue oblique)",
    "batiment_enduit":     "Dalle beton / toit plat (vue oblique)",
}


# =============================================================================
# SPLIT
# =============================================================================

def split_coco(annotations_file, output_dir):
    """Split le COCO JSON en deux sous-datasets : nadir et oblique."""

    print("=" * 65)
    print("   SPLIT DATASET : nadir (Production_*) / oblique (Snapshot_*)")
    print("=" * 65)

    if not os.path.exists(annotations_file):
        raise FileNotFoundError(f"Annotations introuvables : {annotations_file}")

    with open(annotations_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # -----------------------------------------------------------------------
    # Afficher le contenu du dataset source
    # -----------------------------------------------------------------------

    cat_by_name = {c["name"]: c for c in coco["categories"]}
    cat_by_id   = {c["id"]:   c for c in coco["categories"]}

    print(f"\n📋 Dataset source : {annotations_file}")
    print(f"   Images totales : {len(coco['images'])}")
    print(f"   Annotations    : {len(coco['annotations'])}")
    print(f"\n   Categories:")
    for c in sorted(coco["categories"], key=lambda x: x["id"]):
        count = sum(1 for a in coco["annotations"] if a["category_id"] == c["id"])
        print(f"      [{c['id']}] {c['name']:25s}: {count} annotations")

    # Verifier que les classes attendues existent
    missing = [c for c in NADIR_CLASSES + OBLIQUE_CLASSES if c not in cat_by_name]
    if missing:
        print(f"\n   Avertissement — classes manquantes : {missing}")

    # -----------------------------------------------------------------------
    # Separer les images par type
    # -----------------------------------------------------------------------

    nadir_imgs   = [img for img in coco["images"]
                    if Path(img["file_name"]).name.startswith(NADIR_PREFIX)]
    oblique_imgs = [img for img in coco["images"]
                    if Path(img["file_name"]).name.startswith(OBLIQUE_PREFIX)]
    other_imgs   = [img for img in coco["images"]
                    if not Path(img["file_name"]).name.startswith(NADIR_PREFIX)
                    and not Path(img["file_name"]).name.startswith(OBLIQUE_PREFIX)]

    print(f"\n🖼️  Repartition des images:")
    print(f"   Nadir   (Production_*) : {len(nadir_imgs)}")
    print(f"   Oblique (Snapshot_*)   : {len(oblique_imgs)}")
    if other_imgs:
        print(f"   Autres (ignorees)      : {len(other_imgs)}")
        for img in other_imgs[:5]:
            print(f"      - {img['file_name']}")

    # -----------------------------------------------------------------------
    # Filtrer les annotations par image et par classe
    # -----------------------------------------------------------------------

    nadir_cat_ids   = {cat_by_name[c]["id"] for c in NADIR_CLASSES   if c in cat_by_name}
    oblique_cat_ids = {cat_by_name[c]["id"] for c in OBLIQUE_CLASSES if c in cat_by_name}

    nadir_img_ids   = {img["id"] for img in nadir_imgs}
    oblique_img_ids = {img["id"] for img in oblique_imgs}

    nadir_anns = [
        a for a in coco["annotations"]
        if a["image_id"] in nadir_img_ids and a["category_id"] in nadir_cat_ids
    ]
    oblique_anns = [
        a for a in coco["annotations"]
        if a["image_id"] in oblique_img_ids and a["category_id"] in oblique_cat_ids
    ]

    # -----------------------------------------------------------------------
    # Construire les JSON COCO filtres
    # -----------------------------------------------------------------------

    nadir_cats   = [c for c in coco["categories"] if c["name"] in NADIR_CLASSES]
    oblique_cats = [c for c in coco["categories"] if c["name"] in OBLIQUE_CLASSES]

    nadir_json = {
        "info":        coco.get("info", {"description": "Nadir subset - panneau_solaire"}),
        "licenses":    coco.get("licenses", []),
        "images":      nadir_imgs,
        "annotations": nadir_anns,
        "categories":  nadir_cats,
    }
    oblique_json = {
        "info":        coco.get("info", {"description": "Oblique subset - batiments"}),
        "licenses":    coco.get("licenses", []),
        "images":      oblique_imgs,
        "annotations": oblique_anns,
        "categories":  oblique_cats,
    }

    # -----------------------------------------------------------------------
    # Sauvegarder les JSON
    # -----------------------------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    nadir_path   = os.path.join(output_dir, "instances_nadir.json")
    oblique_path = os.path.join(output_dir, "instances_oblique.json")

    with open(nadir_path, "w", encoding="utf-8") as f:
        json.dump(nadir_json, f, indent=2, ensure_ascii=False)

    with open(oblique_path, "w", encoding="utf-8") as f:
        json.dump(oblique_json, f, indent=2, ensure_ascii=False)

    print(f"\n   instances_nadir.json   : {len(nadir_imgs)} images, {len(nadir_anns)} annotations")
    print(f"      -> {nadir_path}")
    print(f"\n   instances_oblique.json : {len(oblique_imgs)} images, {len(oblique_anns)} annotations")
    print(f"      -> {oblique_path}")

    # -----------------------------------------------------------------------
    # Generer les fichiers classes YAML pour YOLO
    # -----------------------------------------------------------------------

    nadir_yaml_path   = _generate_classes_yaml(nadir_cats,   "classes_nadir.yaml")
    oblique_yaml_path = _generate_classes_yaml(oblique_cats, "classes_oblique.yaml")

    return nadir_path, oblique_path, nadir_yaml_path, oblique_yaml_path


# =============================================================================
# CLASSES YAML
# =============================================================================

def _generate_classes_yaml(categories, output_path):
    """Generer un fichier classes YAML compatible avec YOLO et train.py."""

    class_names = [c["name"] for c in sorted(categories, key=lambda x: x["id"])]

    data = {
        "classes":      ["__background__"] + class_names,
        "colors":       {name: COLORS.get(name, [128, 128, 128]) for name in class_names},
        "descriptions": {name: DESCRIPTIONS.get(name, "") for name in class_names},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    print(f"\n   {output_path} : {class_names}")
    return output_path


# =============================================================================
# RESUME
# =============================================================================

def print_split_summary(nadir_path, oblique_path):
    """Afficher un tableau recapitulatif du split."""

    print("\n" + "=" * 65)
    print("   RESUME DU SPLIT")
    print("=" * 65)

    for label, path in [("NADIR", nadir_path), ("OBLIQUE", oblique_path)]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        cat_by_id    = {c["id"]: c["name"] for c in data["categories"]}
        ann_counts   = {}
        for ann in data["annotations"]:
            name = cat_by_id.get(ann["category_id"], "?")
            ann_counts[name] = ann_counts.get(name, 0) + 1

        # Images with at least one annotation
        imgs_with_anns = {a["image_id"] for a in data["annotations"]}

        print(f"\n  [{label}]")
        print(f"   Images totales    : {len(data['images'])}")
        print(f"   Images annotees   : {len(imgs_with_anns)}")
        print(f"   Annotations total : {len(data['annotations'])}")
        for cat_name, count in sorted(ann_counts.items()):
            print(f"   * {cat_name:25s}: {count}")

    print("\n" + "=" * 65)
    print("\n   Etapes suivantes :")
    print("   python train.py --mode nadir")
    print("   python train.py --mode oblique")
    print("   python evaluate_dual.py")
    print("=" * 65)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split COCO dataset en nadir (Production_*) et oblique (Snapshot_*)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--annotations",
        default=os.getenv(
            "DETECTION_DATASET_ANNOTATIONS_FILE",
            "../dataset1/annotations/instances_default.json",
        ),
        help="Fichier annotations COCO source (instances_default.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie pour les JSON splits (defaut: meme dossier que --annotations)",
    )
    args = parser.parse_args()

    # Par defaut: meme dossier que le fichier source
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.annotations))

    nadir_path, oblique_path, _, _ = split_coco(args.annotations, args.output_dir)
    print_split_summary(nadir_path, oblique_path)
