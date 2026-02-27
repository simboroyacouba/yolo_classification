"""
Test YOLO - Prédiction rapide
Configuration: .env + classes.yaml
Usage: python test.py --image image.jpg
"""

import os
import sys
import argparse
import yaml
import json
import numpy as np
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime
import time
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

CONFIG = {
    "model_path": os.path.join(os.getenv("OUTPUT_DIR", "./output"), "best_model.pt"),
    "output_dir": "./test_results",
    "score_threshold": 0.5,
    "classes": load_classes(CLASSES_FILE),
    "colors": load_colors(CLASSES_FILE),
}


def format_time(seconds):
    return f"{seconds*1000:.1f} ms" if seconds < 1 else f"{seconds:.2f} s"


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
        preds['class_names'] = [CONFIG['classes'][int(l) + 1] for l in preds['labels']]
    
    return image, preds


def visualize(image, preds, output_path=None, show=True):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis('off')
    axes[1].imshow(image)
    
    for box, class_name, score in zip(preds['boxes'], preds['class_names'], preds['scores']):
        color = [c/255 for c in CONFIG['colors'].get(class_name, (128,128,128))]
        x1, y1, x2, y2 = box
        axes[1].add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none'))
        axes[1].text(x1, y1-5, f"{class_name}\n{score:.2f}", fontsize=9, color='white',
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    
    axes[1].set_title(f"YOLO: {len(preds['boxes'])} objets | ⏱️ {format_time(preds['inference_time'])}")
    axes[1].axis('off')
    
    legend = [patches.Patch(facecolor=[c/255 for c in col], label=name) for name, col in CONFIG['colors'].items()]
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
        'by_class': {c: {'count': 0} for c in CONFIG['classes'][1:]},
        'detections': []
    }
    
    for i, (box, class_name, score) in enumerate(zip(preds['boxes'], preds['class_names'], preds['scores'])):
        report['by_class'][class_name]['count'] += 1
        report['detections'].append({'id': i, 'class': class_name, 'confidence': float(score), 'bbox': box.tolist()})
    
    return report


def print_report(report):
    print(f"\n{'='*60}")
    print(f"📊 {report['model']} - {report['image']}")
    print(f"{'='*60}")
    print(f"   ⏱️  Temps: {report['inference_time_ms']:.1f} ms")
    print(f"   🎯 Objets: {report['total_objects']}")
    if report['by_class']:
        print("\n   Par classe:")
        for c, data in report['by_class'].items():
            if data['count'] > 0:
                print(f"      • {c}: {data['count']}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Test YOLO")
    parser.add_argument("--model", default=CONFIG["model_path"])
    parser.add_argument("--image", help="Image unique")
    parser.add_argument("--folder", help="Dossier d'images")
    parser.add_argument("--output", default=CONFIG["output_dir"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        print("❌ Spécifiez --image ou --folder")
        sys.exit(1)
    
    print(f"🧠 Modèle: {args.model}")
    model = YOLO(args.model)
    
    os.makedirs(args.output, exist_ok=True)
    
    if args.image:
        images = [Path(args.image)]
    else:
        exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        images = sorted([p for p in Path(args.folder).iterdir() if p.suffix.lower() in exts])
    
    print(f"\n🖼️  {len(images)} image(s)\n")
    
    reports = []
    
    for img_path in images:
        print(f"🔍 {img_path.name}")
        image, preds = predict(model, str(img_path), args.threshold)
        visualize(image, preds, os.path.join(args.output, f"{img_path.stem}_yolo.png"), show=not args.no_display)
        report = generate_report(preds, img_path.name)
        reports.append(report)
        print_report(report)
    
    with open(os.path.join(args.output, 'reports_yolo.json'), 'w') as f:
        json.dump(reports, f, indent=2)
    
    print(f"\n📁 Résultats: {args.output}")


if __name__ == "__main__":
    main()
