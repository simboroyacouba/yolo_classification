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
        return ["__background__", "batiment_peint", "batiment_non_enduit", "batiment_enduit",
                "menuiserie_metallique", "menuiserie_aluminium",
                "cloture_enduit", "cloture_non_enduit", "cloture_peinte"]
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f).get('classes', [])


# =============================================================================
# CONFIG
# =============================================================================

CLASSES_FILE = os.getenv("CLASSES_FILE", "classes.yaml")

CONFIG = {
    "model_path":           os.getenv("MODEL_PATH", None),
    "output_dir":           os.getenv("EVALUATION_DIR", "./evaluation"),
    "dataset_dir":          os.getenv("DATASET_DIR", "./output/dataset"),
    "classes_file":         CLASSES_FILE,
    "nadir_classes_file":   os.getenv("NADIR_CLASSES_FILE",   "classes_nadir.yaml"),
    "oblique_classes_file": os.getenv("OBLIQUE_CLASSES_FILE", "classes_oblique.yaml"),
    "classes":              load_classes(CLASSES_FILE),
    "score_threshold":      0.5,
    "iou_thresholds":       [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
}


def _find_yolo_project_model(project):
    """Trouve best_model.pt dans le projet YOLO ./<project>/train*/."""
    if not os.path.isdir(project):
        return None
    runs = sorted(
        [d for d in os.listdir(project) if d.startswith("train")],
        key=lambda x: int(x[5:]) if x[5:].isdigit() else 0,
        reverse=True,
    )
    for run in runs:
        for fname in ("best_model.pt", "weights/best.pt", "weights/best.pt"):
            candidate = os.path.join(project, run, fname)
            if os.path.exists(candidate):
                return candidate
    return None


def _find_test_info(base_output, sub=None):
    """Retourne le chemin du test_info.json pour le sous-mode donné (None = unified)."""
    if sub:
        candidate = os.path.join(base_output, sub, "dataset", "test_info.json")
    else:
        candidate = os.path.join(base_output, "dataset", "test_info.json")
    if os.path.exists(candidate):
        return candidate
    # Fallback: walk to find any test_info.json
    search_root = os.path.join(base_output, sub) if sub else base_output
    if os.path.isdir(search_root):
        for root, _, files in os.walk(search_root):
            if "test_info.json" in files:
                return os.path.join(root, "test_info.json")
    return None


def auto_discover():
    """
    Decouvre automatiquement les modeles entraines.
    Retourne ('unified', model_path, test_info_path)
          ou ('dual', [(mode, model_path, test_info_path), ...])
          ou None si rien trouve.
    """
    env_model   = CONFIG["model_path"]
    base_output = os.getenv("OUTPUT_DIR", "./output")

    if env_model and os.path.exists(env_model):
        ti = _find_test_info(base_output)
        return ("unified", env_model, ti)

    # Modes unifies (simple, attention, optimize, all)
    for project in ("simple", "attention", "optimize", "all"):
        mp = _find_yolo_project_model(project)
        if mp:
            ti = _find_test_info(base_output)
            return ("unified", mp, ti)

    # Fallback legacy: runs/detect/
    if os.path.exists("runs/detect"):
        candidates = sorted(os.listdir("runs/detect"), reverse=True)
        for folder in candidates:
            for fname in ("best_model.pt", "weights/best.pt"):
                mp = os.path.join("runs/detect", folder, fname)
                if os.path.exists(mp):
                    ti = _find_test_info(base_output)
                    return ("unified", mp, ti)

    # Mode dual (nadir + oblique)
    pairs = []
    for sub_mode in ("nadir", "oblique"):
        mp = _find_yolo_project_model(sub_mode)
        if mp:
            ti = _find_test_info(base_output, sub_mode)
            pairs.append((sub_mode, mp, ti))

    if pairs:
        return ("dual", pairs)

    return None


def _load_test_set(test_info_path):
    """Charge test_images_dir et test_labels_dir depuis test_info.json."""
    if test_info_path is None:
        return None, None
    with open(test_info_path, "r") as f:
        info = json.load(f)
    return info.get("test_images_dir"), info.get("test_labels_dir")


def _run_single_eval(model_path, test_images_dir, test_labels_dir, class_names, label=""):
    """Evalue un modele YOLO sur un test set. Retourne le dict results ou None."""
    if not os.path.exists(str(model_path)):
        print(f"   Modele introuvable: {model_path}")
        return None
    if not os.path.isdir(str(test_images_dir or "")):
        print(f"   Dossier images introuvable: {test_images_dir}")
        return None
    if not os.path.isdir(str(test_labels_dir or "")):
        test_labels_dir = str(test_images_dir).replace("/images/", "/labels/")
    if not os.path.isdir(test_labels_dir):
        print(f"   Dossier labels introuvable: {test_labels_dir}")
        return None

    print(f"   Modele:       {model_path}")
    print(f"   Test images:  {test_images_dir}")

    gt = load_yolo_labels(test_labels_dir, test_images_dir)
    if not gt:
        print("   Aucune image dans le test set!")
        return None

    model = YOLO(model_path)
    class_names_no_bg = [c for c in class_names if c != "__background__"]
    calc = MetricsCalculator(len(class_names_no_bg), class_names, CONFIG["iou_thresholds"])

    for img_file in tqdm(gt.keys(), desc=f"Eval {label}"):
        img_path = os.path.join(test_images_dir, img_file)
        if not os.path.exists(img_path):
            continue
        res = model.predict(img_path, conf=CONFIG["score_threshold"], verbose=False)[0]
        g   = gt[img_file]
        if len(res.boxes) > 0:
            pb = res.boxes.xyxy.cpu().numpy()
            pl = res.boxes.cls.cpu().numpy().astype(int)
            ps = res.boxes.conf.cpu().numpy()
        else:
            pb = np.zeros((0, 4)); pl = np.zeros((0,), dtype=int); ps = np.zeros((0,))
        calc.add_image(pb, pl, ps, g["boxes"], g["labels"])

    return calc.compute(), len(gt)


def _merge_yolo_results(entries):
    """
    Fusionne des resultats de modes differents.
    entries: [(results_dict, mode_label), ...]
    """
    merged_per_class  = {}
    merged_map_class  = {}
    iou_thresholds    = CONFIG["iou_thresholds"]

    for results, _ in entries:
        merged_per_class.update(results["per_class"])
        merged_map_class.update(results["mAP_per_class"])

    all_names = list(merged_per_class.keys())
    overall   = {}
    for t in iou_thresholds:
        tp = sum(merged_per_class[n][f"iou_{t}"]["TP"] for n in all_names)
        fp = sum(merged_per_class[n][f"iou_{t}"]["FP"] for n in all_names)
        fn = sum(merged_per_class[n][f"iou_{t}"]["FN"] for n in all_names)
        p  = tp / (tp + fp) if tp + fp > 0 else 0
        r  = tp / (tp + fn) if tp + fn > 0 else 0
        overall[f"iou_{t}"] = {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": p, "Recall": r,
            "F1": 2 * p * r / (p + r) if p + r > 0 else 0,
        }

    return {
        "per_class":      merged_per_class,
        "mAP_per_class":  merged_map_class,
        "overall":        overall,
        "mAP50":    float(np.mean([v["AP50"]    for v in merged_map_class.values()])),
        "mAP50_95": float(np.mean([v["AP50_95"] for v in merged_map_class.values()])),
        "macro_avg": {
            "Precision": float(np.mean([merged_per_class[n]["iou_0.5"]["Precision"] for n in all_names])),
            "Recall":    float(np.mean([merged_per_class[n]["iou_0.5"]["Recall"]    for n in all_names])),
            "F1":        float(np.mean([merged_per_class[n]["iou_0.5"]["F1"]        for n in all_names])),
        },
    }


def _print_save_yolo_results(results, output_dir, n_images, model_path, label=""):
    os.makedirs(output_dir, exist_ok=True)
    title = f"RESULTATS TEST SET{' — ' + label if label else ''}"
    print("\n" + "=" * 70)
    print(f"   {title}")
    print("=" * 70)
    print(f"   Images testees: {n_images}")
    ma = results.get("macro_avg", results["overall"]["iou_0.5"])
    print(f"   mAP@50:    {results['mAP50']:.4f} ({results['mAP50']*100:.2f}%)")
    print(f"   mAP@50:95: {results['mAP50_95']:.4f}")
    print(f"   Precision: {ma['Precision']:.4f}")
    print(f"   Recall:    {ma['Recall']:.4f}")
    print(f"   F1-Score:  {ma['F1']:.4f}")
    print("=" * 70)
    if results["mAP_per_class"]:
        print("\n   Par classe (IoU=0.5):")
        for name in results["mAP_per_class"]:
            m = results["per_class"][name]["iou_0.5"]
            print(f"   {name:<25} P={m['Precision']:.3f} R={m['Recall']:.3f} F1={m['F1']:.3f}")

    with open(os.path.join(output_dir, "metrics_test_set.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    plot_metrics(results, output_dir)

    with open(os.path.join(output_dir, "evaluation_report_test_set.txt"), "w", encoding="utf-8") as f:
        f.write(f"EVALUATION YOLO - TEST SET - {datetime.now()}\n{'='*50}\n\n")
        f.write(f"Images testees: {n_images}\nModele: {model_path}\n\n")
        f.write(f"mAP@50: {results['mAP50']:.4f}\nmAP@50:95: {results['mAP50_95']:.4f}\n")
        f.write(f"Precision: {ma['Precision']:.4f}\nRecall: {ma['Recall']:.4f}\nF1-Score: {ma['F1']:.4f}\n")
        if results["mAP_per_class"]:
            f.write("\nPAR CLASSE (IoU=0.5)\n" + "-" * 50 + "\n")
            for name in results["mAP_per_class"]:
                m = results["per_class"][name]["iou_0.5"]
                f.write(f"{name}: P={m['Precision']:.4f} R={m['Recall']:.4f} F1={m['F1']:.4f}\n")

    print(f"\n   Resultats sauvegardes: {output_dir}")


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
        
        results['mAP_per_class'] = {}
        for class_id, name in enumerate(class_names_no_bg):
            results['mAP_per_class'][name] = {
                'AP50':    float(results['per_class'][name]['iou_0.5']['Precision']),
                'AP50_95': float(np.mean([results['per_class'][name][f'iou_{t}']['Precision']
                                          for t in self.iou_thresholds]))
            }

        # ✅ mAP50 / mAP50:95 = macro-moyenne des AP par classe (correct)
        results['mAP50']    = float(np.mean([results['mAP_per_class'][n]['AP50']    for n in class_names_no_bg]))
        results['mAP50_95'] = float(np.mean([results['mAP_per_class'][n]['AP50_95'] for n in class_names_no_bg]))

        # ✅ Métriques globales = macro-moyenne sur les classes (P, R, F1 cohérents)
        results['macro_avg'] = {
            'Precision': float(np.mean([results['per_class'][n]['iou_0.5']['Precision'] for n in class_names_no_bg])),
            'Recall':    float(np.mean([results['per_class'][n]['iou_0.5']['Recall']    for n in class_names_no_bg])),
            'F1':        float(np.mean([results['per_class'][n]['iou_0.5']['F1']        for n in class_names_no_bg])),
        }
        # overall['iou_0.5'] conservé pour compatibilité (micro-average TP/FP agrégé)

        if self.all_ious:
            results['iou_stats'] = {
                'mean':   float(np.mean(self.all_ious)),
                'median': float(np.median(self.all_ious))
            }

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
    print("   EVALUATION YOLO - TEST SET (10%)")
    print("=" * 70)

    discovery = auto_discover()
    if discovery is None:
        print("   Aucun modele trouve! Lancez d'abord train.py")
        return

    if discovery[0] == "unified":
        _, model_path, test_info_path = discovery
        test_images_dir, test_labels_dir = _load_test_set(test_info_path)

        # Fallback standard si test_info absent
        if test_images_dir is None:
            for p in (CONFIG["dataset_dir"] + "/images/test",
                      "./output/dataset/images/test"):
                if os.path.isdir(p):
                    test_images_dir = p
                    test_labels_dir = p.replace("/images/", "/labels/")
                    break

        ret = _run_single_eval(model_path, test_images_dir, test_labels_dir,
                               CONFIG["classes"], label="all")
        if ret is None:
            return
        results, n_images = ret
        results["evaluation_info"] = {
            "dataset": "TEST SET (10%)", "num_images": n_images,
            "model_path": str(model_path), "timestamp": datetime.now().isoformat(),
        }
        _print_save_yolo_results(results, CONFIG["output_dir"], n_images, model_path)

    else:  # dual
        _, pairs = discovery
        entries = []

        for sub_mode, model_path, test_info_path in pairs:
            print(f"\n{'─'*50}")
            print(f"   [{sub_mode.upper()}]")
            print(f"{'─'*50}")
            test_images_dir, test_labels_dir = _load_test_set(test_info_path)
            # Determine classes for this sub_mode
            sub_classes_file = (CONFIG.get("nadir_classes_file", CLASSES_FILE)
                                if sub_mode == "nadir"
                                else CONFIG.get("oblique_classes_file", CLASSES_FILE))
            sub_classes = load_classes(sub_classes_file)

            ret = _run_single_eval(model_path, test_images_dir, test_labels_dir,
                                   sub_classes, label=sub_mode)
            if ret is None:
                print(f"   Mode {sub_mode} ignore.")
                continue
            results, n_images = ret
            sub_out = os.path.join(CONFIG["output_dir"], sub_mode)
            _print_save_yolo_results(results, sub_out, n_images, model_path, label=sub_mode.upper())
            entries.append((results, sub_mode, n_images))

        if len(entries) > 1:
            merged = _merge_yolo_results([(r, m) for r, m, _ in entries])
            total  = sum(n for _, _, n in entries)
            paths  = " + ".join(m for _, m, _ in entries)
            merged["evaluation_info"] = {
                "dataset": "TEST SET (10%) — GLOBAL",
                "num_images": total, "model_path": paths,
                "timestamp": datetime.now().isoformat(),
            }
            _print_save_yolo_results(merged, os.path.join(CONFIG["output_dir"], "global"),
                                     total, paths, label="GLOBAL (NADIR + OBLIQUE)")


if __name__ == "__main__":
    main()