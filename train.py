"""
Entraînement YOLO pour détection des toitures cadastrales
Dataset: Images aériennes annotées avec CVAT (format COCO)
Classes: Chargées depuis classes.yaml
Configuration: Chargée depuis .env
"""

import os
import json
import yaml
import shutil
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import time
import gc
import torch
import csv
import warnings
warnings.filterwarnings('ignore')

# Charger .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optionnel


# =============================================================================
# CHARGEMENT DES CLASSES
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    """Charger les classes depuis YAML (sans __background__ pour YOLO)"""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Fichier introuvable: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    classes = [c for c in data.get('classes', []) if c != '__background__']
    
    print(f"📋 Classes chargées depuis {yaml_path}:")
    for i, c in enumerate(classes):
        print(f"   [{i}] {c}")
    
    return classes


# =============================================================================
# CONFIGURATION (depuis .env)
# =============================================================================

CONFIG = {
    # Chemins
    "images_dir": os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
    "annotations_file": os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"),
    "test_images_dir": os.getenv("DETECTION_TEST_IMAGES_DIR", "../test"),
    "output_dir": os.getenv("OUTPUT_DIR", "./output"),
    "classes_file": os.getenv("CLASSES_FILE", "classes.yaml"),
    
    # Classes
    "classes": None,
    
    # Modèle YOLO
    "model_version": os.getenv("YOLO_VERSION", "yolo26"),
    "model_size": os.getenv("YOLO_SIZE", "n"),
    
    # Hyperparamètres
    "num_epochs": int(os.getenv("NUM_EPOCHS", "25")),
    "batch_size": int(os.getenv("BATCH_SIZE", "2")),
    "learning_rate": float(os.getenv("LEARNING_RATE", "0.005")),
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "image_size": int(os.getenv("IMAGE_SIZE", "640")),
    "train_split": float(os.getenv("TRAIN_SPLIT", "0.85")),
    "save_every": 5,
}


# =============================================================================
# UTILITAIRES
# =============================================================================

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds//60)}m {int(seconds%60)}s"
    else:
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"


def coco_to_yolo_bbox(bbox, img_width, img_height):
    """Convertir bbox COCO -> YOLO format normalisé"""
    x, y, w, h = bbox
    x_center = max(0, min(1, (x + w / 2) / img_width))
    y_center = max(0, min(1, (y + h / 2) / img_height))
    width = max(0, min(1, w / img_width))
    height = max(0, min(1, h / img_height))
    return [x_center, y_center, width, height]


def stratified_split(coco, train_split=0.85, seed=42):
    """
    Split stratifié pour garantir que chaque classe est représentée
    proportionnellement dans train et val.
    
    Algorithme:
    1. Pour chaque image, déterminer la classe "dominante" (la plus fréquente)
    2. Grouper les images par classe dominante
    3. Pour chaque groupe, prendre 85% pour train et 15% pour val
    4. Fusionner les résultats
    
    Retourne: train_ids, val_ids, stats
    """
    np.random.seed(seed)
    
    # Déterminer la classe dominante de chaque image
    image_dominant_class = {}
    image_all_classes = {}  # Pour les stats
    
    for img_id in coco.imgs:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        if not anns:
            image_dominant_class[img_id] = -1
            image_all_classes[img_id] = []
            continue
        
        # Compter les classes dans cette image
        class_counts = {}
        for ann in anns:
            cat_id = ann['category_id']
            class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
        
        dominant_class = max(class_counts, key=class_counts.get)
        image_dominant_class[img_id] = dominant_class
        image_all_classes[img_id] = list(class_counts.keys())
    
    # Grouper les images par classe dominante
    images_by_class = {}
    for img_id, class_id in image_dominant_class.items():
        if class_id not in images_by_class:
            images_by_class[class_id] = []
        images_by_class[class_id].append(img_id)
    
    # Split stratifié
    train_ids = []
    val_ids = []
    
    for class_id, img_ids in images_by_class.items():
        np.random.shuffle(img_ids)
        split_idx = int(len(img_ids) * train_split)
        
        # Garantir au moins 1 image dans val si possible
        if split_idx == len(img_ids) and len(img_ids) > 1:
            split_idx = len(img_ids) - 1
        
        train_ids.extend(img_ids[:split_idx])
        val_ids.extend(img_ids[split_idx:])
    
    # Mélanger les résultats finaux
    np.random.shuffle(train_ids)
    np.random.shuffle(val_ids)
    
    # Calculer les statistiques de distribution
    stats = {'train': {}, 'val': {}}
    
    for cat_id in coco.getCatIds():
        stats['train'][cat_id] = 0
        stats['val'][cat_id] = 0
    
    for img_id in train_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['train'][ann['category_id']] = stats['train'].get(ann['category_id'], 0) + 1
    
    for img_id in val_ids:
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            stats['val'][ann['category_id']] = stats['val'].get(ann['category_id'], 0) + 1
    
    return train_ids, val_ids, stats


def print_split_stats(coco, stats):
    """Afficher les statistiques de distribution des classes"""
    print("\n   📊 Distribution des classes (split stratifié):")
    print(f"   {'Classe':<25} {'Train':>8} {'Val':>8} {'Total':>8} {'Val%':>8}")
    print(f"   {'-'*57}")
    
    for cat_id in coco.getCatIds():
        cat_name = coco.cats[cat_id]['name']
        train_count = stats['train'].get(cat_id, 0)
        val_count = stats['val'].get(cat_id, 0)
        total = train_count + val_count
        val_pct = (val_count / total * 100) if total > 0 else 0
        
        # Alerte si déséquilibre
        status = "⚠️" if val_count == 0 else "✅"
        print(f"   {cat_name:<25} {train_count:>8} {val_count:>8} {total:>8} {val_pct:>7.1f}% {status}")
    
    print(f"   {'-'*57}")


def prepare_yolo_dataset(images_dir, annotations_file, output_dir, classes, train_split=0.85):
    """Convertir le dataset COCO en format YOLO avec split stratifié"""
    
    print("📂 Préparation du dataset YOLO...")
    
    dataset_dir = os.path.join(output_dir, "dataset")
    dirs = {
        'train_img': os.path.join(dataset_dir, "images", "train"),
        'val_img': os.path.join(dataset_dir, "images", "val"),
        'train_lbl': os.path.join(dataset_dir, "labels", "train"),
        'val_lbl': os.path.join(dataset_dir, "labels", "val"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    coco = COCO(annotations_file)
    cat_ids = coco.getCatIds()
    cat_mapping = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    
    print(f"   Catégories: {[coco.cats[c]['name'] for c in cat_ids]}")
    
    # Split stratifié (au lieu du split naïf)
    train_ids, val_ids, split_stats = stratified_split(coco, train_split, seed=42)
    
    print(f"   Train: {len(train_ids)} images | Val: {len(val_ids)} images")
    
    # Afficher les stats de distribution
    print_split_stats(coco, split_stats)
    
    splits = {
        'train': (train_ids, dirs['train_img'], dirs['train_lbl']),
        'val': (val_ids, dirs['val_img'], dirs['val_lbl']),
    }
    
    stats = {'train': 0, 'val': 0, 'annotations': 0, 'per_class': {c: 0 for c in classes}}
    
    for split_name, (img_ids, img_dir, lbl_dir) in splits.items():
        for img_id in img_ids:
            img_info = coco.imgs[img_id]
            src = os.path.join(images_dir, img_info['file_name'])
            if not os.path.exists(src):
                continue
            
            shutil.copy2(src, os.path.join(img_dir, img_info['file_name']))
            
            anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_info['file_name'])[0] + '.txt')
            
            with open(lbl_path, 'w') as f:
                for ann in anns:
                    if ann.get('iscrowd', 0):
                        continue
                    class_id = cat_mapping.get(ann['category_id'])
                    bbox = ann.get('bbox')
                    if class_id is None or not bbox or bbox[2] <= 0 or bbox[3] <= 0:
                        continue
                    
                    yolo_bbox = coco_to_yolo_bbox(bbox, img_info['width'], img_info['height'])
                    f.write(f"{class_id} {' '.join(f'{v:.6f}' for v in yolo_bbox)}\n")
                    stats['annotations'] += 1
                    if class_id < len(classes):
                        stats['per_class'][classes[class_id]] += 1
            
            stats[split_name] += 1
    
    # dataset.yaml
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump({
            'path': os.path.abspath(dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'names': {i: name for i, name in enumerate(classes)}
        }, f, default_flow_style=False)
    
    print(f"   Annotations: {stats['annotations']}")
    return yaml_path, stats


# =============================================================================
# ENTRAÎNEMENT
# =============================================================================

def train_yolo():
    """Entraîner YOLO"""
    
    CONFIG["classes"] = load_classes(CONFIG["classes_file"])
    
    print("=" * 70)
    print(f"   YOLO ({CONFIG['model_version']}{CONFIG['model_size']}) - Détection des Toitures")
    print("=" * 70)
    print(f"\n📋 CONFIG (.env)")
    print(f"   Images:      {CONFIG['images_dir']}")
    print(f"   Annotations: {CONFIG['annotations_file']}")
    print(f"   Modèle:      {CONFIG['model_version']}{CONFIG['model_size']}")
    print(f"   Epochs:      {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']} | LR: {CONFIG['learning_rate']}")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    yaml_path, dataset_stats = prepare_yolo_dataset(
        CONFIG["images_dir"], CONFIG["annotations_file"],
        CONFIG["output_dir"], CONFIG["classes"], CONFIG["train_split"]
    )
    
    model_name = f"{CONFIG['model_version']}{CONFIG['model_size']}.pt"
    print(f"\n🧠 Chargement: {model_name}")
    
    gc.collect()
    model = YOLO(model_name)
    
    print("\n" + "=" * 70)
    print(f"   🚀 ENTRAÎNEMENT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Nettoyer le dossier train précédent pour éviter les conflits
    train_dir = os.path.join(CONFIG["output_dir"], "train")
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    
    results = model.train(
        data=yaml_path,
        epochs=CONFIG["num_epochs"],
        batch=CONFIG["batch_size"],
        imgsz=CONFIG["image_size"],
        lr0=CONFIG["learning_rate"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"],
        project=CONFIG["output_dir"],
        name="train",
        exist_ok=True,
        seed=42,
        verbose=True,
        save=True,
        save_period=CONFIG["save_every"],
        plots=True,
        cache=False,
        workers=0,
    )
    
    total_time = time.time() - start_time
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Trouver le dossier d'entraînement réel (YOLO peut créer train, train2, train3...)
    train_dir = None
    weights_dir = None
    
    # Chercher le dossier le plus récent qui contient weights/best.pt
    for folder in sorted(os.listdir(CONFIG["output_dir"]), reverse=True):
        potential_dir = os.path.join(CONFIG["output_dir"], folder)
        potential_weights = os.path.join(potential_dir, "weights", "best.pt")
        if os.path.isdir(potential_dir) and os.path.exists(potential_weights):
            train_dir = potential_dir
            weights_dir = os.path.join(potential_dir, "weights")
            break
    
    if train_dir is None:
        train_dir = os.path.join(CONFIG["output_dir"], "train")
        weights_dir = os.path.join(train_dir, "weights")
    
    print(f"\n📁 Dossier d'entraînement: {train_dir}")
    
    # Récupérer métriques
    history = {'mAP50': [], 'mAP50_95': [], 'precision': [], 'recall': [],
               'train_box_loss': [], 'train_cls_loss': [], 'train_dfl_loss': [],
               'val_box_loss': [], 'val_cls_loss': [], 'val_dfl_loss': []}
    
    results_csv = os.path.join(train_dir, "results.csv")
    if os.path.exists(results_csv):
        with open(results_csv, 'r') as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}
                for key, col in [('mAP50', 'metrics/mAP50(B)'), ('mAP50_95', 'metrics/mAP50-95(B)'),
                                 ('precision', 'metrics/precision(B)'), ('recall', 'metrics/recall(B)'),
                                 ('train_box_loss', 'train/box_loss'), ('train_cls_loss', 'train/cls_loss'),
                                 ('train_dfl_loss', 'train/dfl_loss'), ('val_box_loss', 'val/box_loss'),
                                 ('val_cls_loss', 'val/cls_loss'), ('val_dfl_loss', 'val/dfl_loss')]:
                    try:
                        history[key].append(float(row.get(col, 0) or 0))
                    except:
                        pass
    
    history['time_stats'] = {
        'total_time': total_time,
        'total_time_formatted': format_time(total_time),
        'avg_epoch_time_formatted': format_time(total_time / CONFIG["num_epochs"]),
    }
    history['config'] = CONFIG
    history['dataset_stats'] = dataset_stats
    
    with open(os.path.join(CONFIG["output_dir"], "history.json"), 'w') as f:
        json.dump(history, f, indent=2, default=str)
    
    # Copier modèles vers la racine de output/
    print("\n📦 Copie des modèles...")
    models_copied = []
    
    for src, dst in [("best.pt", "best_model.pt"), ("last.pt", "final_model.pt")]:
        src_path = os.path.join(weights_dir, src)
        dst_path = os.path.join(CONFIG["output_dir"], dst)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            models_copied.append(dst)
            print(f"   ✅ {dst} ({os.path.getsize(dst_path) / 1024 / 1024:.1f} MB)")
        else:
            print(f"   ⚠️ {src} non trouvé dans {weights_dir}")
    
    if not models_copied:
        print("   ❌ Aucun modèle trouvé!")
        # Lister le contenu pour debug
        print(f"\n   Contenu de {CONFIG['output_dir']}:")
        for item in os.listdir(CONFIG["output_dir"]):
            item_path = os.path.join(CONFIG["output_dir"], item)
            if os.path.isdir(item_path):
                print(f"      📁 {item}/")
                for sub in os.listdir(item_path)[:5]:
                    print(f"         - {sub}")
    
    # Rapport
    best = {k: max(history[k]) if history[k] else 0 for k in ['mAP50', 'mAP50_95', 'precision', 'recall']}
    
    print("\n" + "=" * 70)
    print("   🎉 TERMINÉ")
    print("=" * 70)
    print(f"   mAP@50:     {best['mAP50']:.4f} ({best['mAP50']*100:.2f}%)")
    print(f"   mAP@50:95:  {best['mAP50_95']:.4f}")
    print(f"   Precision:  {best['precision']:.4f}")
    print(f"   Recall:     {best['recall']:.4f}")
    print(f"   ⏱️  Temps:    {format_time(total_time)}")
    print("=" * 70)
    
    # Rapport texte
    with open(os.path.join(CONFIG["output_dir"], "training_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"YOLO ({CONFIG['model_version']}{CONFIG['model_size']}) - Rapport\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {CONFIG['images_dir']}\n")
        f.write(f"Classes: {CONFIG['classes']}\n")
        f.write(f"Epochs: {CONFIG['num_epochs']} | Batch: {CONFIG['batch_size']}\n\n")
        f.write(f"mAP@50: {best['mAP50']:.4f}\nmAP@50:95: {best['mAP50_95']:.4f}\n")
        f.write(f"Precision: {best['precision']:.4f}\nRecall: {best['recall']:.4f}\n")
        f.write(f"Temps: {format_time(total_time)}\n")
    
    # Graphiques
    if history['mAP50']:
        epochs = range(1, len(history['mAP50']) + 1)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        train_loss = [a+b+c for a,b,c in zip(history['train_box_loss'], history['train_cls_loss'], history['train_dfl_loss'])]
        val_loss = [a+b+c for a,b,c in zip(history['val_box_loss'], history['val_cls_loss'], history['val_dfl_loss'])]
        axes[0,0].plot(epochs, train_loss, 'b-', label='Train')
        axes[0,0].plot(epochs, val_loss, 'r-', label='Val')
        axes[0,0].set_title('Loss'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(epochs, history['mAP50'], 'g-', label='mAP@50')
        axes[0,1].plot(epochs, history['mAP50_95'], 'b-', label='mAP@50:95')
        axes[0,1].set_title('mAP'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3); axes[0,1].set_ylim(0,1)
        
        axes[1,0].plot(epochs, history['precision'], 'g-', label='Precision')
        axes[1,0].plot(epochs, history['recall'], 'b-', label='Recall')
        axes[1,0].set_title('Precision/Recall'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3); axes[1,0].set_ylim(0,1)
        
        axes[1,1].plot(epochs, history['train_box_loss'], label='Box')
        axes[1,1].plot(epochs, history['train_cls_loss'], label='Cls')
        axes[1,1].plot(epochs, history['train_dfl_loss'], label='DFL')
        axes[1,1].set_title('Loss Components'); axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["output_dir"], 'training_curves.png'), dpi=150)
        plt.close()
    
    return model, history


if __name__ == "__main__":
    train_yolo()
