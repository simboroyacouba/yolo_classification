"""
Inférence YOLO - Détection des toitures cadastrales
Configuration: .env + classes.yaml
"""

import os
import json
import yaml
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =============================================================================
# CONFIG
# =============================================================================

def load_classes(yaml_path="classes.yaml"):
    if not os.path.exists(yaml_path):
        return ["__background__", "toiture_tole_ondulee", "toiture_tole_bac", "toiture_tuile", "toiture_dalle"]
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f).get('classes', [])


def load_colors(yaml_path="classes.yaml"):
    default = {"toiture_tole_ondulee": (255,0,0), "toiture_tole_bac": (0,255,0), 
               "toiture_tuile": (0,0,255), "toiture_dalle": (255,165,0)}
    if not os.path.exists(yaml_path):
        return default
    with open(yaml_path, 'r', encoding='utf-8') as f:
        colors = yaml.safe_load(f).get('colors', {})
    return {k: tuple(v) for k, v in colors.items()} if colors else default


CLASSES_FILE = os.getenv("CLASSES_FILE", "classes.yaml")
CLASSES = load_classes(CLASSES_FILE)
COLORS = load_colors(CLASSES_FILE)


def format_time(seconds):
    return f"{seconds*1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


# =============================================================================
# INFERENCE
# =============================================================================

def predict(model, image_path, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    
    start = time.time()
    results = model.predict(image_path, conf=threshold, verbose=False)
    inference_time = time.time() - start
    
    result = results[0]
    preds = {'boxes': [], 'labels': [], 'scores': [], 'class_names': [], 'inference_time': inference_time}
    
    if len(result.boxes) > 0:
        preds['boxes'] = result.boxes.xyxy.cpu().numpy()
        preds['labels'] = result.boxes.cls.cpu().numpy().astype(int)
        preds['scores'] = result.boxes.conf.cpu().numpy()
        preds['class_names'] = [CLASSES[int(l) + 1] for l in preds['labels']]
    
    return image, preds


def visualize(image, preds, output_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(image)
    
    for i, (box, class_name, score) in enumerate(zip(preds['boxes'], preds['class_names'], preds['scores'])):
        color = [c/255 for c in COLORS.get(class_name, (128,128,128))]
        x1, y1, x2, y2 = box
        axes[1].add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none'))
        axes[1].text(x1, y1-5, f"{class_name}\n{score:.2f}", fontsize=8, color='white',
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    axes[1].set_title(f"YOLO: {len(preds['boxes'])} objets | ⏱️ {format_time(preds['inference_time'])}")
    axes[1].axis('off')
    
    legend = [patches.Patch(facecolor=[c/255 for c in col], label=name) for name, col in COLORS.items()]
    fig.legend(handles=legend, loc='lower center', ncol=4)
    
    plt.tight_layout(); plt.subplots_adjust(bottom=0.1)
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def generate_report(preds, image_name):
    report = {
        'image': image_name, 'model': 'YOLO', 'timestamp': datetime.now().isoformat(),
        'inference_time_ms': preds['inference_time'] * 1000,
        'total_objects': len(preds['boxes']),
        'by_class': {c: {'count': 0} for c in CLASSES[1:]},
        'detections': []
    }
    
    for i, (box, class_name, score) in enumerate(zip(preds['boxes'], preds['class_names'], preds['scores'])):
        report['by_class'][class_name]['count'] += 1
        report['detections'].append({'id': i, 'class': class_name, 'confidence': float(score), 'bbox': box.tolist()})
    
    return report


def generate_summary(reports, output_dir, total_time):
    summary = {
        'model': 'YOLO', 'timestamp': datetime.now().isoformat(),
        'total_images': len(reports), 'total_time_s': total_time,
        'avg_inference_ms': sum(r['inference_time_ms'] for r in reports) / len(reports) if reports else 0,
        'total_objects': sum(r['total_objects'] for r in reports),
        'by_class': {c: sum(r['by_class'][c]['count'] for r in reports) for c in CLASSES[1:]},
        'per_image': [{'image': r['image'], 'objects': r['total_objects'], 'time_ms': r['inference_time_ms']} for r in reports]
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    with open(os.path.join(output_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"RÉSUMÉ YOLO\n{'='*50}\n")
        f.write(f"Images: {summary['total_images']} | Temps: {total_time:.1f}s\n")
        f.write(f"Temps moyen: {summary['avg_inference_ms']:.1f} ms | Objets: {summary['total_objects']}\n")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Inférence YOLO")
    parser.add_argument("--model", default=os.path.join(os.getenv("OUTPUT_DIR", "./output"), "best_model.pt"))
    parser.add_argument("--input", default=os.getenv("DETECTION_TEST_IMAGES_DIR", "../test"))
    parser.add_argument("--output", default=os.getenv("PREDICTIONS_DIR", "./predictions"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    
    print(f"🧠 Modèle: {args.model}")
    model = YOLO(args.model)
    
    os.makedirs(args.output, exist_ok=True)
    
    input_path = Path(args.input)
    images = sorted([p for p in input_path.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.tif','.tiff'}]) if input_path.is_dir() else [input_path]
    
    print(f"🖼️  {len(images)} image(s)\n")
    
    reports = []
    start_total = time.time()
    
    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] 🔍 {img_path.name}")
        image, preds = predict(model, str(img_path), args.threshold)
        visualize(image, preds, os.path.join(args.output, f"{img_path.stem}_yolo.png"), show=not args.no_display)
        report = generate_report(preds, img_path.name)
        reports.append(report)
        print(f"   ✅ {report['total_objects']} objets | ⏱️ {report['inference_time_ms']:.1f} ms")
    
    with open(os.path.join(args.output, 'reports.json'), 'w') as f:
        json.dump(reports, f, indent=2)
    
    if len(images) > 1:
        summary = generate_summary(reports, args.output, time.time() - start_total)
        print(f"\n📊 Résumé: {summary['total_objects']} objets | {summary['avg_inference_ms']:.1f} ms/image")
    
    print(f"\n📁 Résultats: {args.output}")


if __name__ == "__main__":
    main()
