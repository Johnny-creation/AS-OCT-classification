# AS-OCT Classification and Evaluation Toolkit

This repository provides an end-to-end workflow for anterior segment OCT (AS-OCT) multi-class classification. It covers data
preparation, backbone training, prediction export, rigorous patient-level evaluation, statistical comparisons, calibration,
stacking-based ensembling, and Grad-CAM visualisation.

The reference task is a four-class problem (`0 = Normal`, `1 = Cataract`, `2 = PACG`, `3 = PACG + Cataract`) but the tooling is
implemented generically for `N` classes when possible.

## Key Features

- **Patient-level safety** – data splitting, aggregation, and bootstrap resampling operate at the patient level to prevent data
  leakage.
- **Reusable training loop** – train any supported torchvision backbone with early stopping and reproducible settings.
- **Prediction export** – produce aligned CSV/Parquet tables with logits and probabilities ready for downstream evaluation.
- **Statistics & calibration** – compute balanced accuracy, F1, PR/ROC curves, bootstrap confidence intervals, McNemar tests,
  and temperature scaling.
- **Stacking ensembles** – leak-free meta-learning pipeline with Logistic Regression, SVM, Random Forest, KNN, GaussianNB, and
  optional XGBoost (if installed).
- **Interpretability** – lightweight Grad-CAM script for qualitative inspection of trained backbones.

## Repository Layout

```
AS-OCT-classification/
├── dataset/                     # JSON manifests produced by split.py
├── examples/                    # Synthetic predictions and demo script
├── reports/                     # Generated metrics, figures, and models
├── scripts/                     # CLI utilities (evaluation, calibration, stacking)
├── src/
│   ├── metrics/                 # Patient-level metrics, calibration, statistics
│   ├── models/factory.py        # Backbone construction helpers
│   ├── plots/                   # Matplotlib helpers for PR/ROC/CM
│   ├── stacking/                # Leak-free stacking pipeline
│   └── utils/                   # IO helpers for prediction tables
├── dataset_utils.py             # Dataset loaders exposing metadata
├── generate_predictions.py      # Export logits/probabilities for trained models
├── heatmap_visualization.py     # Grad-CAM visualisation script
├── split.py                     # Patient-level data splitting (JSON manifests)
├── test_multimodel.py           # Convenience patient-level evaluator
├── train_multimodel.py          # Backbone training entry-point
├── README.md                    # This document
├── README_eval.md               # Legacy evaluation notes (superseded by README.md)
└── requirements.txt             # Python dependencies
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Runtime Dependencies

- Python 3.8+
- PyTorch 2.0+ with torchvision
- scikit-learn, pandas, numpy, matplotlib, tqdm, joblib
- `pytorch-grad-cam` for heatmap generation
- Optional: `xgboost` (for stacking) – the pipeline will skip it if unavailable.

## Data Preparation

1. Organise images under `data/<CLASS>/<PATIENT_ID>/*.jpg` (or similar). Patient IDs should be encoded in folder names and can
   include `_OS/_OD` suffixes.
2. Generate patient-level splits using:

   ```bash
   python split.py
   ```

   The script writes JSON manifests to `dataset/` (train, validation, ensemble, test). Each entry contains the image path,
   label, and patient identifier.

## Backbone Training

Train one or more backbones with a single command:

```bash
python train_multimodel.py \
  --models resnet50 densenet169 efficientnet_b4 \
  --train-json dataset/asoct.train-model.json \
  --val-json dataset/asoct.val-model.json \
  --epochs 50 --batch-size 32 --patience 7
```

Key options:

- `--models`: identifiers from `src/models/factory.py` (e.g. `resnet50`, `resnext50`, `efficientnet_b4`, `vgg16`, `convnext_tiny`,
  `mobilenet_v2`, `densenet169`).
- `--no-pretrained`: disable ImageNet initialisation.
- `--weights-dir`: where `best_<model>.pth` checkpoints are saved.

Early stopping is based on patient-level validation accuracy. The best state per model is stored in `weights/` by default.

## Prediction Export

After training, export logits and probabilities for any manifest (e.g. ensemble set, test set):

```bash
python generate_predictions.py \
  --dataset dataset/asoct.val-ensemble.json \
  --models resnet50 efficientnet_b4 \
  --weights-dir weights \
  --out reports/predictions/val_ensemble.csv
```

The resulting table includes columns:

- `patient_id`, `image_id`, `image_path`, `label_name`, `y_true`
- `<model>_logit_<k>` and `<model>_proba_<k>` for each class `k`

These files feed directly into the evaluation and stacking scripts.

## Patient-level Evaluation

### Quick evaluation

`test_multimodel.py` combines prediction export and patient-level metrics in one step:

```bash
python test_multimodel.py \
  --dataset dataset/asoct.test.json \
  --models resnet50 efficientnet_b4 \
  --weights-dir weights \
  --outdir reports/evaluation \
  --agg mean
```

Outputs:

- `image_level_predictions.csv`: logits and probabilities for each image
- `<model>_patient_level.csv`: aggregated patient-level probabilities
- `patient_metrics.csv`: balanced accuracy, macro/micro F1, per-class AP & ROC-AUC, and macro means

### Detailed toolkit (scripts/)

The `scripts/` directory contains CLI utilities that operate on prediction tables (CSV or Parquet). Examples use synthetic data
from `examples/data/`.

```bash
# Patient-level metrics, bootstrap CI, and plots
python scripts/eval_patient_level.py \
  --input examples/data/preds_resnet50.csv examples/data/preds_vit_b.csv \
  --id-col patient_id \
  --true-col y_true \
  --proba-prefixes resnet50,vit_b \
  --agg mean \
  --bootstrap 2000 \
  --outdir reports

# Model comparison (McNemar + AUC difference)
python scripts/compare_models.py \
  --inputs examples/data/preds_resnet50.csv examples/data/preds_vit_b.csv \
  --id-col patient_id \
  --true-col y_true \
  --proba-prefixes resnet50,vit_b \
  --outdir reports

# Temperature scaling on ensemble validation set
python scripts/fit_temperature.py \
  --ens-file examples/data/ens_resnet50.csv \
  --apply-to examples/data/preds_resnet50.csv \
  --by per_model \
  --outdir reports

# Stacking meta-learner (leak-free)
python scripts/train_stacking.py \
  --ens-file examples/data/stack_ens_features.csv \
  --labels-col y_true \
  --feature-type logit \
  --meta lr \
  --outdir reports
python scripts/infer_stacking.py \
  --stacker reports/stacking_lr.pkl \
  --inputs examples/data/preds_resnet50.csv examples/data/preds_vit_b.csv \
  --feature-type logit \
  --out reports/stacked_preds.csv
```

All evaluation figures (PR, ROC, confusion matrix, reliability diagrams) are written to `reports/figs/` with timestamps.

## Statistical Testing & Confidence Intervals

The evaluation scripts support bootstrap resampling (default 2000 iterations) using patients as the sampling unit. Reported
metrics include confidence intervals (2.5% – 97.5%). For pairwise comparisons `scripts/compare_models.py` provides:

- **McNemar test** for paired accuracy differences
- **AUC difference** via DeLong (if available) or bootstrap fallback

Significance is assessed at `α = 0.05`.

## Calibration & Temperature Scaling

`scripts/fit_temperature.py` fits positive temperature parameters on the disjoint ensemble validation set (`D_ens`). The script
supports per-model or ensemble-wide scaling and writes calibrated predictions plus a JSON summary of fitted temperatures. The
calibration module also computes Brier score, ECE, ACE, and reliability diagrams with confidence bands.

## Stacking Ensembles

The stacking pipeline (`scripts/train_stacking.py` / `scripts/infer_stacking.py`) ensures no leakage by consuming only frozen
predictions on `D_ens`. Supported meta-learners:

- Logistic Regression (default)
- Linear SVM + Platt scaling (`CalibratedClassifierCV`)
- RandomForestClassifier
- KNeighborsClassifier
- GaussianNB
- XGBoost (optional, skipped when not installed)

Feature inputs can be softmax probabilities or raw logits. Additional ensemble baselines (weighted average, hard voting) are
available directly through `src/stacking/pipeline.py`.

## Grad-CAM Visualisation

Generate qualitative heatmaps for trained models:

```bash
python heatmap_visualization.py \
  --image path/to/sample.jpg \
  --model resnet50 \
  --weights weights/best_resnet50.pth \
  --output reports/figs/sample_gradcam.png
```

Specify `--target-class` to visualise a particular class index. If omitted, the predicted class is used.

## Demo Workflow

Run the synthetic example pipeline end-to-end:

```bash
bash examples/run_demo.sh
```

This script demonstrates aggregation, evaluation, temperature scaling, stacking, and statistical comparison using the example
tables in `examples/data/`.

## Testing

Unit tests cover aggregation, calibration, stacking, and dataset handling:

```bash
pytest -q
```

At least ten assertions check patient-level aggregation, bootstrap CIs, McNemar statistics, temperature scaling behaviour, and
leakage safeguards in the stacking pipeline.

## Troubleshooting

- **Missing probability columns**: ensure exported predictions follow the `<prefix>_proba_<class>` naming convention.
- **Probability rows not summing to one**: utilities automatically renormalise and log warnings.
- **Unavailable dependencies**: optional components (e.g. XGBoost) degrade gracefully; install extra packages if needed.

For additional details on evaluation tooling refer to `README_eval.md` (kept for historical context).
