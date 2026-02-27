# YOLO - Détection des Toitures Cadastrales

Détection d'objets avec YOLO pour la classification des types de toitures.
**Structure identique à Mask R-CNN / Faster R-CNN pour comparaison équitable.**

## Structure du projet

```
yolo_detection/
├── train.py              # Entraînement
├── evaluate.py           # Évaluation (mAP, Precision, Recall)
├── inference.py          # Inférence sur nouvelles images
├── test.py               # Test rapide
├── verify_dataset.py     # Vérification du dataset
├── requirements.txt      # Dépendances
├── README.md
├── output/               # Modèles entraînés
│   ├── best_model.pt
│   ├── final_model.pt
│   ├── history.json
│   ├── training_curves.png
│   └── training_report.txt
├── evaluation/           # Résultats d'évaluation
│   ├── metrics.json
│   ├── metrics_per_class.png
│   └── evaluation_report.txt
└── predictions/          # Prédictions
    ├── *_yolo.png
    └── reports.json
```

## Versions YOLO supportées

| Version | Modèle | Année |
|---------|--------|-------|
| YOLOv8 | `yolov8n.pt` | 2023 |
| YOLOv9 | `yolov9n.pt` | 2024 |
| YOLOv10 | `yolov10n.pt` | 2024 |
| YOLOv11 | `yolo11n.pt` | 2024 |

Tailles: `n` (nano), `s` (small), `m` (medium), `l` (large), `x` (xlarge)

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Vérifier le dataset

```bash
python verify_dataset.py --images chemin/images --annotations chemin/annotations.json
```

### 2. Entraîner

```bash
python train.py
```

Modifier `CONFIG` dans `train.py` :
```python
CONFIG = {
    "images_dir": "chemin/vers/images",
    "annotations_file": "chemin/vers/annotations.json",
    "model_version": "yolo11",  # ou yolov8, yolov9, yolov10
    "model_size": "n",          # n, s, m, l, x
    "num_epochs": 25,
    "batch_size": 2,
    ...
}
```

### 3. Évaluer

```bash
python evaluate.py
```

### 4. Inférence

```bash
# Une image
python inference.py --model output/best_model.pt --input image.jpg

# Un dossier
python inference.py --model output/best_model.pt --input dossier/ --no-display
```

### 5. Test rapide

```bash
python test.py --image image.jpg
python test.py --folder dossier/
```

## Hyperparamètres

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `num_epochs` | 25 | Nombre d'epochs |
| `batch_size` | 2 | Taille du batch |
| `learning_rate` | 0.005 | Taux d'apprentissage |
| `momentum` | 0.9 | Momentum SGD |
| `weight_decay` | 0.0005 | Régularisation L2 |
| `image_size` | 640 | Taille des images |
| `train_split` | 0.85 | 85% train / 15% val |

## Métriques

| Métrique | Description |
|----------|-------------|
| mAP@50 | Mean Average Precision à IoU=0.5 |
| mAP@50:95 | Moyenne des AP de 0.5 à 0.95 |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × (P × R) / (P + R) |

## Fichiers générés

### Après entraînement (`output/`)
- `best_model.pt` - Meilleur modèle
- `final_model.pt` - Modèle final
- `history.json` - Historique des métriques
- `training_curves.png` - Courbes d'entraînement
- `training_report.txt` - Rapport textuel

### Après évaluation (`evaluation/`)
- `metrics.json` - Métriques détaillées
- `metrics_per_class.png` - Graphiques par classe
- `evaluation_report.txt` - Rapport

### Après inférence (`predictions/`)
- `*_yolo.png` - Visualisations
- `reports.json` - Rapports détaillés
- `summary.json` / `summary.txt` - Résumé global

## Comparaison avec autres modèles

| Aspect | YOLO | Faster R-CNN | SSD |
|--------|------|--------------|-----|
| Vitesse | ⚡ Très rapide | 🐢 Lent | ⚡ Rapide |
| Précision | ✅ Bonne | ✅ Très bonne | ⚠️ Moyenne |
| Petits objets | ⚠️ Moyen | ✅ Bon | ❌ Faible |

## Auteur

Projet de thèse - Exploitation de l'IA pour l'évaluation cadastrale
Burkina Faso - SYCAD/DGI
