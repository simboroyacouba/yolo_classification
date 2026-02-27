"""
Évaluation YOLO - Détection des toitures cadastrales
Configuration: .env + classes.yaml
"""

import os
import json
import yaml
import numpy as np
from ultralytics import YOLO
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime
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
    "images_dir": os.getenv("DETECTION_DATASET_IMAGES_DIR", "../dataset1/images/default"),
    "annotations_file": os.getenv("DETECTION_DATASET_ANNOTATIONS_FILE", "../dataset1/annotations/instances_default.json"),
    "model_path": os.path.join(os.getenv("OUTPUT_DIR", "./output"), "best_model.pt"),
    "output_dir": os.getenv("EVALUATION_DIR", "./evaluation"),
    "classes_file": CLASSES_FILE,
    "classes": load_classes(CLASSES_FILE),
    "score_threshold": 0.5,
    "iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
}


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
            for class_id in range(1, self.num_classes):
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
        
        for class_id in range(1, self.num_classes):
            name = self.class_names[class_id]
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
            tp = sum(self.tp[c][t] for c in range(1, self.num_classes))
            fp = sum(self.fp[c][t] for c in range(1, self.num_classes))
            fn = sum(self.fn[c][t] for c in range(1, self.num_classes))
            p = tp/(tp+fp) if tp+fp > 0 else 0
            r = tp/(tp+fn) if tp+fn > 0 else 0
            results['overall'][f'iou_{t}'] = {
                'TP': tp, 'FP': fp, 'FN': fn,
                'Precision': p, 'Recall': r, 'F1': 2*p*r/(p+r) if p+r > 0 else 0
            }
        
        results['mAP50'] = results['overall']['iou_0.5']['Precision']
        results['mAP50_95'] = np.mean([results['overall'][f'iou_{t}']['Precision'] for t in self.iou_thresholds])
        
        results['mAP_per_class'] = {}
        for class_id in range(1, self.num_classes):
            name = self.class_names[class_id]
            results['mAP_per_class'][name] = {
                'AP50': results['per_class'][name]['iou_0.5']['Precision'],
                'AP50_95': np.mean([results['per_class'][name][f'iou_{t}']['Precision'] for t in self.iou_thresholds])
            }
        
        if self.all_ious:
            results['iou_stats'] = {'mean': float(np.mean(self.all_ious)), 'median': float(np.median(self.all_ious))}
        
        return results


def load_ground_truths(images_dir, annotations_file):
    coco = COCO(annotations_file)
    cat_mapping = {cat_id: idx + 1 for idx, cat_id in enumerate(coco.getCatIds())}
    
    gt = {}
    for img_id, img_info in coco.imgs.items():
        boxes, labels = [], []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            if ann.get('iscrowd', 0):
                continue
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x+w, y+h])
                labels.append(cat_mapping[ann['category_id']])
        gt[img_info['file_name']] = {
            'boxes': np.array(boxes) if boxes else np.zeros((0, 4)),
            'labels': np.array(labels) if labels else np.zeros((0,), dtype=int)
        }
    return gt


def plot_metrics(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    class_names = list(results['mAP_per_class'].keys())
    x = np.arange(len(class_names))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(x - 0.2, [results['mAP_per_class'][c]['AP50'] for c in class_names], 0.4, label='AP@50')
    axes[0].bar(x + 0.2, [results['mAP_per_class'][c]['AP50_95'] for c in class_names], 0.4, label='AP@50:95')
    axes[0].set_xticks(x); axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_title('AP par classe'); axes[0].legend(); axes[0].set_ylim(0, 1)
    
    w = 0.25
    axes[1].bar(x-w, [results['per_class'][c]['iou_0.5']['Precision'] for c in class_names], w, label='Precision')
    axes[1].bar(x, [results['per_class'][c]['iou_0.5']['Recall'] for c in class_names], w, label='Recall')
    axes[1].bar(x+w, [results['per_class'][c]['iou_0.5']['F1'] for c in class_names], w, label='F1')
    axes[1].set_xticks(x); axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_title('P/R/F1 (IoU=0.5)'); axes[1].legend(); axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_per_class.png'), dpi=150)
    plt.close()


def main():
    print("=" * 70)
    print("   ÉVALUATION YOLO")
    print("=" * 70)
    
    print(f"\n📋 Config: {CONFIG['images_dir']}")
    print(f"   Modèle: {CONFIG['model_path']}")
    
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    gt = load_ground_truths(CONFIG["images_dir"], CONFIG["annotations_file"])
    print(f"   {len(gt)} images")
    
    model = YOLO(CONFIG["model_path"])
    calc = MetricsCalculator(len(CONFIG["classes"]), CONFIG["classes"], CONFIG["iou_thresholds"])
    
    for img_file in tqdm(gt.keys(), desc="Évaluation"):
        img_path = os.path.join(CONFIG["images_dir"], img_file)
        if not os.path.exists(img_path):
            continue
        
        res = model.predict(img_path, conf=CONFIG["score_threshold"], verbose=False)[0]
        g = gt[img_file]
        
        if len(res.boxes) > 0:
            pred_boxes = res.boxes.xyxy.cpu().numpy()
            pred_labels = res.boxes.cls.cpu().numpy().astype(int) + 1
            pred_scores = res.boxes.conf.cpu().numpy()
        else:
            pred_boxes, pred_labels, pred_scores = np.zeros((0,4)), np.zeros((0,), dtype=int), np.zeros((0,))
        
        calc.add_image(pred_boxes, pred_labels, pred_scores, g['boxes'], g['labels'])
    
    results = calc.compute()
    
    print("\n" + "=" * 70)
    print(f"   mAP@50:    {results['mAP50']:.4f}")
    print(f"   mAP@50:95: {results['mAP50_95']:.4f}")
    print(f"   Precision: {results['overall']['iou_0.5']['Precision']:.4f}")
    print(f"   Recall:    {results['overall']['iou_0.5']['Recall']:.4f}")
    print("=" * 70)
    
    with open(os.path.join(CONFIG["output_dir"], "metrics.json"), 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    plot_metrics(results, CONFIG["output_dir"])
    
    with open(os.path.join(CONFIG["output_dir"], "evaluation_report.txt"), 'w', encoding='utf-8') as f:
        f.write(f"ÉVALUATION YOLO - {datetime.now()}\n")
        f.write("=" * 50 + "\n")
        f.write(f"mAP@50: {results['mAP50']:.4f}\nmAP@50:95: {results['mAP50_95']:.4f}\n")
        f.write(f"Precision: {results['overall']['iou_0.5']['Precision']:.4f}\n")
        f.write(f"Recall: {results['overall']['iou_0.5']['Recall']:.4f}\n")


if __name__ == "__main__":
    main()
