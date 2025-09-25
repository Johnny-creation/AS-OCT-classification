#!/bin/bash
set -euo pipefail

OUTDIR="reports"
mkdir -p "$OUTDIR"

python scripts/eval_patient_level.py \
  --input examples/data/preds_resnet50.csv examples/data/preds_vit_b.csv \
  --id-col patient_id \
  --true-col y_true \
  --proba-prefixes resnet50,vit_b \
  --agg mean \
  --bootstrap 100 \
  --outdir "$OUTDIR"

python scripts/fit_temperature.py \
  --ens-file examples/data/ens_resnet50.csv \
  --apply-to examples/data/preds_resnet50.csv \
  --by per_model \
  --outdir "$OUTDIR"

python scripts/train_stacking.py \
  --ens-file examples/data/stack_ens_features.csv \
  --labels-col y_true \
  --feature-type logit \
  --meta lr \
  --outdir "$OUTDIR"

python scripts/infer_stacking.py \
  --stacker "$OUTDIR"/stacking_lr.pkl \
  --inputs examples/data/preds_resnet50.csv examples/data/preds_vit_b.csv \
  --feature-type logit \
  --out "$OUTDIR"/stacked_preds.csv

python scripts/compare_models.py \
  --inputs examples/data/preds_resnet50.csv examples/data/preds_vit_b.csv \
  --id-col patient_id \
  --true-col y_true \
  --proba-prefixes resnet50,vit_b \
  --outdir "$OUTDIR"
