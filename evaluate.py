"""
Évaluation YOLO - Détection des toitures cadastrales
Évaluation sur le TEST SET (10% du dataset)
Configuration: .env + classes.yaml
"""

import os
import json
import yaml
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CLASSES
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    if not os.path.exists(yaml_path):
        return ["__background__", "toiture_tole_ondulee", "toiture_tole_bac", "toiture_tuile", "toiture_dalle"]
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f).get('classes', [])


# =============================================================================
# CONFIG
# =============================================================================

CLASSES_FILE = os.getenv("CLASSES_FILE", "classes.yaml")

CONFIG = {
    "model_path": os.getenv("MODEL_PATH", None),  # Sera détecté automatiquement
    "output_dir": os.getenv("EVALUATION_DIR", "./evaluation"),
    "dataset_dir": os.getenv("DATASET_DIR", "./output/dataset"),
    "classes_file": CLASSES_FILE,
    "classes": load_classes(CLASSES_FILE),
    "score_threshold": 0.5,
    "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
}


def find_model_and_test_set():
    """Trouver automatiquement le modèle et le test set"""
    
    model_path = CONFIG["model_path"]
    test_images_dir = None
    test_labels_dir = None
    
    # Chercher le modèle
    if model_path is None or not os.path.exists(str(model_path)):
        # Chercher dans runs/detect/
        for root, dirs, files in os.walk("runs/detect"):
            if "best.pt" in files:
                model_path = os.path.join(root, "best.pt")
                break
            if "best_model.pt" in files:
                model_path = os.path.join(root, "best_model.pt")
                break
        
        # Chercher aussi dans output/
        if model_path is None:
            for root, dirs, files in os.walk("output"):
                if "best.pt" in files:
                    model_path = os.path.join(root, "best.pt")
                    break
                if "best_model.pt" in files:
                    model_path = os.path.join(root, "best_model.pt")
                    break
    
    # Chercher le test set
    # 1. Chercher test_info.json
    for root, dirs, files in os.walk("."):
        if "test_info.json" in files:
            with open(os.path.join(root, "test_info.json"), 'r') as f:
                info = json.load(f)
                test_images_dir = info.get('test_images_dir')
                test_labels_dir = info.get('test_labels_dir')
            break
    
    # 2. Sinon chercher dans output/dataset/images/test
    if test_images_dir is None:
        possible_paths = [
            "./output/dataset/images/test",
            "output/dataset/images/test",
            CONFIG["dataset_dir"] + "/images/test",
        ]
        for p in possible_paths:
            if os.path.exists(p):
                test_images_dir = p
                test_labels_dir = p.replace("/images/", "/labels/")
                break
    
    return model_path, test_images_dir, test_labels_dir


# =============================================================================
# MÉTRIQUES
# =============================================================================

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter) if (area1 + area2 - inter) > 0 else 0


class MetricsCalculator:
    def __init__(self, num_classes, class_names, iou_thresholds):
        self.num_classes = num_classes
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds
        self.tp = defaultdict(lambda: defaultdict(int))
        self.fp = defaultdict(lambda: defaultdict(int))
        self.fn = defaultdict(lambda: defaultdict(int))
        self.all_ious = []
    
    def add_image(self, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels):
        for iou_thresh in self.iou_thresholds:
            for class_id in range(0, self.num_classes):
                p_mask, g_mask = pred_labels == class_id, gt_labels == class_id
                p_boxes, p_scores = pred_boxes[p_mask], pred_scores[p_mask]
                g_boxes = gt_boxes[g_mask]
                
                if len(g_boxes) == 0 and len(p_boxes) == 0:
                    continue
                if len(g_boxes) == 0:
                    self.fp[class_id][iou_thresh] += len(p_boxes)
                    continue
                if len(p_boxes) == 0:
                    self.fn[class_id][iou_thresh] += len(g_boxes)
                    continue
                
                iou_matrix = np.array([[calculate_iou(p, g) for g in g_boxes] for p in p_boxes])
                if iou_thresh == 0.5:
                    self.all_ious.extend(iou_matrix.flatten().tolist())
                
                matched = set()
                for i in np.argsort(-p_scores):
                    best_j = -1
                    for j in range(len(g_boxes)):
                        if j not in matched and iou_matrix[i, j] >= iou_thresh:
                            if best_j < 0 or iou_matrix[i, j] > iou_matrix[i, best_j]:
                                best_j = j
                    if best_j >= 0:
                        matched.add(best_j)
                        self.tp[class_id][iou_thresh] += 1
                    else:
                        self.fp[class_id][iou_thresh] += 1
                self.fn[class_id][iou_thresh] += len(g_boxes) - len(matched)
    
    def compute(self):
        results = {'per_class': {}, 'overall': {}}
        
        # Classes sans background (YOLO utilise indices 0-based)
        class_names_no_bg = [c for c in self.class_names if c != '__background__']
        
        for class_id, name in enumerate(class_names_no_bg):
            results['per_class'][name] = {}
            for t in self.iou_thresholds:
                tp, fp, fn = self.tp[class_id][t], self.fp[class_id][t], self.fn[class_id][t]
                p = tp/(tp+fp) if tp+fp > 0 else 0
                r = tp/(tp+fn) if tp+fn > 0 else 0
                results['per_class'][name][f'iou_{t}'] = {
                    'TP': tp, 'FP': fp, 'FN': fn,
                    'Precision': p, 'Recall': r, 'F1': 2*p*r/(p+r) if p+r > 0 else 0
                }
        
        for t in self.iou_thresholds:
            tp = sum(self.tp[c][t] for c in range(len(class_names_no_bg)))
            fp = sum(self.fp[c][t] for c in range(len(class_names_no_bg)))
            fn = sum(self.fn[c][t] for c in range(len(class_names_no_bg)))
            p = tp/(tp+fp) if tp+fp > 0 else 0
            r = tp/(tp+fn) if tp+fn > 0 else 0
            results['overall'][f'iou_{t}'] = {
                'TP': tp, 'FP': fp, 'FN': fn,
                'Precision': p, 'Recall': r, 'F1': 2*p*r/(p+r) if p+r > 0 else 0
            }
        
        results['mAP50'] = results['overall']['iou_0.5']['Precision']
        results['mAP50_95'] = np.mean([results['overall'][f'iou_{t}']['Precision'] for t in self.iou_thresholds])
        
        results['mAP_per_class'] = {}
        for class_id, name in enumerate(class_names_no_bg):
            results['mAP_per_class'][name] = {
                'AP50': results['per_class'][name]['iou_0.5']['Precision'],
                'AP50_95': np.mean([results['per_class'][name][f'iou_{t}']['Precision'] for t in self.iou_thresholds])
            }
        
        if self.all_ious:
            results['iou_stats'] = {'mean': float(np.mean(self.all_ious)), 'median': float(np.median(self.all_ious))}
        
        return results


def load_yolo_labels(labels_dir, images_dir):
    """Charger les labels YOLO du test set"""
    gt = {}
    
    for img_file in os.listdir(images_dir):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
            continue
        
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            # Obtenir les dimensions de l'image
            img_path = os.path.join(images_dir, img_file)
            img = Image.open(img_path)
            img_w, img_h = img.size
            
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])
                        
                        # Convertir YOLO -> xyxy
                        x1 = (x_center - w/2) * img_w
                        y1 = (y_center - h/2) * img_h
                        x2 = (x_center + w/2) * img_w
                        y2 = (y_center + h/2) * img_h
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id)
        
        gt[img_file] = {
            'boxes': np.array(boxes) if boxes else np.zeros((0, 4)),
            'labels': np.array(labels) if labels else np.zeros((0,), dtype=int)
        }
    
    return gt


def plot_metrics(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = list(results['mAP_per_class'].keys())
    if not class_names:
        return
    
    x = np.arange(len(class_names))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(x - 0.2, [results['mAP_per_class'][c]['AP50'] for c in class_names], 0.4, label='AP@50')
    axes[0].bar(x + 0.2, [results['mAP_per_class'][c]['AP50_95'] for c in class_names], 0.4, label='AP@50:95')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_title('AP par classe (TEST SET)')
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    
    w = 0.25
    axes[1].bar(x-w, [results['per_class'][c]['iou_0.5']['Precision'] for c in class_names], w, label='Precision')
    axes[1].bar(x, [results['per_class'][c]['iou_0.5']['Recall'] for c in class_names], w, label='Recall')
    axes[1].bar(x+w, [results['per_class'][c]['iou_0.5']['F1'] for c in class_names], w, label='F1')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_title('P/R/F1 (IoU=0.5) - TEST SET')
    axes[1].legend()
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_test_set.png'), dpi=150)
    plt.close()


def main():
    print("=" * 70)
    print("   ÉVALUATION YOLO - TEST SET (10%)")
    print("=" * 70)
    
    # Trouver le modèle et le test set
    model_path, test_images_dir, test_labels_dir = find_model_and_test_set()
    
    if model_path is None or not os.path.exists(str(model_path)):
        print("❌ Modèle non trouvé!")
        print("   Spécifiez MODEL_PATH ou lancez d'abord train.py")
        return
    
    if test_images_dir is None or not os.path.exists(str(test_images_dir)):
        print("❌ Test set non trouvé!")
        print("   Lancez d'abord train.py pour créer le split 70/20/10")
        return
    
    print(f"\n📋 Configuration:")
    print(f"   Modèle: {model_path}")
    print(f"   Test images: {test_images_dir}")
    print(f"   Test labels: {test_labels_dir}")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # Charger les ground truths du test set (format YOLO)
    print("\n📂 Chargement du test set...")
    gt = load_yolo_labels(test_labels_dir, test_images_dir)
    print(f"   {len(gt)} images de test")
    
    if len(gt) == 0:
        print("❌ Aucune image trouvée dans le test set!")
        return
    
    # Charger le modèle
    print(f"\n🧠 Chargement du modèle...")
    model = YOLO(model_path)
    
    # Classes sans background
    class_names_no_bg = [c for c in CONFIG["classes"] if c != '__background__']
    calc = MetricsCalculator(len(class_names_no_bg), CONFIG["classes"], CONFIG["iou_thresholds"])
    
    # Évaluation sur le test set
    print("\n📊 Évaluation sur le TEST SET...")
    for img_file in tqdm(gt.keys(), desc="Test"):
        img_path = os.path.join(test_images_dir, img_file)
        if not os.path.exists(img_path):
            continue
        
        res = model.predict(img_path, conf=CONFIG["score_threshold"], verbose=False)[0]
        g = gt[img_file]
        
        if len(res.boxes) > 0:
            pred_boxes = res.boxes.xyxy.cpu().numpy()
            pred_labels = res.boxes.cls.cpu().numpy().astype(int)
            pred_scores = res.boxes.conf.cpu().numpy()
        else:
            pred_boxes, pred_labels, pred_scores = np.zeros((0,4)), np.zeros((0,), dtype=int), np.zeros((0,))
        
        calc.add_image(pred_boxes, pred_labels, pred_scores, g['boxes'], g['labels'])
    
    results = calc.compute()
    
    # Ajouter les infos sur le test set
    results['evaluation_info'] = {
        'dataset': 'TEST SET (10%)',
        'num_images': len(gt),
        'model_path': str(model_path),
        'timestamp': datetime.now().isoformat()
    }
    
    print("\n" + "=" * 70)
    print("   📊 RÉSULTATS SUR LE TEST SET")
    print("=" * 70)
    print(f"   Images testées: {len(gt)}")
    print(f"   mAP@50:    {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)")
    print(f"   mAP@50:95: {results['mAP50_95']:.4f}")
    print(f"   Precision: {results['overall']['iou_0.5']['Precision']:.4f}")
    print(f"   Recall:    {results['overall']['iou_0.5']['Recall']:.4f}")
    print(f"   F1-Score:  {results['overall']['iou_0.5']['F1']:.4f}")
    print("=" * 70)
    
    # Par classe
    if results['mAP_per_class']:
        print("\n   Par classe (IoU=0.5):")
        for name in results['mAP_per_class']:
            m = results['per_class'][name]['iou_0.5']
            print(f"   {name:<25} P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f}")
    
    # Sauvegarder
    with open(os.path.join(CONFIG["output_dir"], "metrics_test_set.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    plot_metrics(results, CONFIG["output_dir"])
    
    # Rapport
    with open(os.path.join(CONFIG["output_dir"], "evaluation_report_test_set.txt"), 'w', encoding='utf-8') as f:
        f.write(f"ÉVALUATION YOLO - TEST SET - {datetime.now()}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Images testées: {len(gt)}\n")
        f.write(f"Modèle: {model_path}\n\n")
        f.write(f"mAP@50: {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)\n")
        f.write(f"mAP@50:95: {results['mAP50_95']:.4f}\n")
        f.write(f"Precision: {results['overall']['iou_0.5']['Precision']:.4f}\n")
        f.write(f"Recall: {results['overall']['iou_0.5']['Recall']:.4f}\n")
        f.write(f"F1-Score: {results['overall']['iou_0.5']['F1']:.4f}\n")
        
        if results['mAP_per_class']:
            f.write("\n\nPAR CLASSE (IoU=0.5)\n")
            f.write("-" * 50 + "\n")
            for name in results['mAP_per_class']:
                m = results['per_class'][name]['iou_0.5']
                f.write(f"{name}: P={m['Precision']:.4f} R={m['Recall']:.4f} F1={m['F1']:.4f}\n")
    
    print(f"\n📁 Résultats sauvegardés: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()