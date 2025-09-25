# Patient-level Evaluation and Stacking Toolkit

> **Note:** This document is kept for historical context. The primary project overview and up-to-date instructions live in
> [`README.md`](README.md).

This repository contains tooling for rigorous patient-level evaluation of AS-OCT multi-class classification models. The utilities focus on data leakage prevention, statistical testing, calibration, and stacking-based ensembling.

## Data format
Prediction tables are expected to be CSV or Parquet files containing the following columns:

- `patient_id`: patient identifier (string or integer)
- `y_true`: ground-truth class label (`0` = Normal, `1` = Cataract, `2` = PACG, `3` = PACG+Cataract)
- `proba_*` columns: probability for each class. For multiple models, prefix the columns (e.g. `resnet50_proba_0`).
- Optional `logit_*` columns: raw logits for each class, following the same prefix convention.
- Additional metadata columns (e.g. `image_id`) are preserved.

When multiple files are supplied the CLI tools automatically align them using shared identifier columns (patient/image ID). Files are validated to ensure that probability rows sum to one (within tolerance); degenerate rows are replaced with uniform distributions and warnings are logged.

## Command-line tools

### Patient-level evaluation
```bash
python scripts/eval_patient_level.py \
  --input examples/data/preds_resnet50.csv \
  --id-col patient_id \
  --true-col y_true \
  --proba-prefixes resnet50 \
  --agg mean \
  --bootstrap 2000 \
  --outdir reports
```

Outputs:
- Patient-level aggregated predictions (`reports/resnet50/resnet50_patient_level.csv`)
- Metrics summary (`reports/patient_metrics.csv`) and bootstrap 95% CI (`reports/patient_metrics_ci.csv`)
- Figures under `reports/figs/`: PR curves, ROC curves, confusion matrix, reliability diagram (timestamped filenames)

### Model comparison (McNemar + AUC difference)
```bash
python scripts/compare_models.py \
  --inputs examples/data/preds_resnet50.csv examples/data/preds_vit_b.csv \
  --id-col patient_id \
  --true-col y_true \
  --proba-prefixes resnet50,vit_b \
  --outdir reports
```

Produces `reports/model_comparisons.csv` with McNemar statistics, p-values, and bootstrap AUC difference confidence intervals.

### Temperature scaling
```bash
python scripts/fit_temperature.py \
  --ens-file examples/data/ens_resnet50.csv \
  --apply-to examples/data/preds_resnet50.csv \
  --by per_model \
  --outdir reports
```

Outputs `reports/temperatures.json` with fitted temperature(s) and `reports/calibrated_predictions.csv` containing updated probabilities.

### Stacking meta-learner
```bash
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

The trained stacker is saved to `reports/stacking_lr.pkl`. Inference produces a fused probability table with columns `proba_0` – `proba_3` (plus identifiers if available).

## Statistical outputs

- **Bootstrap confidence intervals**: computed at the patient level with default 2000 resamples; reproducible via `--seed`.
- **McNemar test**: identifies statistically significant differences in patient-level accuracy (alpha=0.05).
- **AUC difference**: attempts DeLong analysis if available, otherwise falls back to bootstrap with explicit logging.
- **Calibration metrics**: Brier score, ECE, ACE, and reliability diagrams with diagonal reference lines.

## Examples

A convenience script runs the full toolchain on synthetic data:

```bash
bash examples/run_demo.sh
```

Generated artifacts appear under `reports/`, illustrating the full workflow from aggregation to stacking and comparison. Confidence interval tables list lower/upper bounds; values crossing zero (for differences) or excluding key thresholds indicate statistical significance.

## Testing

Run unit tests to validate the tooling:

```bash
pytest -q
```

Tests cover patient-level aggregation, bootstrap CI computation, McNemar statistics, calibration (including positive temperatures), and stacking training/inference safeguards.
