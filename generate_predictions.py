"""Generate prediction tables from trained backbones."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_utils import ASOCTDatasetJSON
from src.models import available_models, create_model
from src.utils.io import normalise_probabilities, save_dataframe


LOGGER = logging.getLogger("generate_predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export image-level predictions to CSV/Parquet")
    parser.add_argument(
        "--dataset",
        default="dataset/asoct.val-ensemble.json",
        help="JSON manifest containing inference samples",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50"],
        help="Backbone identifiers to load. Available: %(choices)s" % {"choices": ", ".join(available_models())},
    )
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="Directory containing fine-tuned weight files (best_<model>.pth)",
    )
    parser.add_argument(
        "--out",
        default="reports/predictions/predictions.csv",
        help="Output CSV/Parquet path",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument(
        "--skip-logits",
        action="store_true",
        help="Do not store raw logits in the exported table",
    )
    return parser.parse_args()


def _build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _prepare_base_records(dataset: ASOCTDatasetJSON) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    for path, label_idx, meta in dataset.samples:
        record = {
            "patient_id": meta.patient_id,
            "image_id": meta.image_id,
            "image_path": meta.path,
            "label_name": meta.label,
            "y_true": int(label_idx),
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


def _load_weights(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Missing weight file: {path}")
    return torch.load(path, map_location=device)


def _extract_inputs(batch):
    if len(batch) == 3:
        inputs, _, _ = batch
    elif len(batch) == 2:
        inputs, _ = batch
    else:
        raise ValueError(f"Unexpected batch format: {batch}")
    return inputs


def _predict_for_model(
    model_name: str,
    dataloader: DataLoader,
    num_classes: int,
    weights_path: Path,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    state_dict = _load_weights(weights_path, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = _extract_inputs(batch).to(device)
            logits = model(inputs)
            logits_list.append(logits.cpu())

    logits_tensor = torch.cat(logits_list, dim=0)
    probas = F.softmax(logits_tensor, dim=1).numpy()
    probas = normalise_probabilities(probas)
    return logits_tensor.numpy(), probas


def run_inference(
    dataset_path: str,
    model_names: Iterable[str],
    weights_dir: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    include_logits: bool,
) -> pd.DataFrame:
    dataset = ASOCTDatasetJSON(dataset_path, _build_transforms())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    base_df = _prepare_base_records(dataset)
    num_classes = len(dataset.classes)

    for model_name in model_names:
        if model_name not in available_models():
            raise ValueError(f"Unsupported model: {model_name}")
        weights_path = weights_dir / f"best_{model_name}.pth"
        LOGGER.info("Running inference for %s", model_name)
        logits, probas = _predict_for_model(
            model_name=model_name,
            dataloader=dataloader,
            num_classes=num_classes,
            weights_path=weights_path,
            device=device,
        )
        for cls_idx in range(num_classes):
            if include_logits:
                base_df[f"{model_name}_logit_{cls_idx}"] = logits[:, cls_idx]
            base_df[f"{model_name}_proba_{cls_idx}"] = probas[:, cls_idx]
    return base_df


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    device = torch.device(args.device)
    predictions = run_inference(
        dataset_path=args.dataset,
        model_names=args.models,
        weights_dir=Path(args.weights_dir),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        include_logits=not args.skip_logits,
    )

    output_path = Path(args.out)
    save_dataframe(predictions, output_path)
    LOGGER.info("Predictions saved to %s", output_path)


if __name__ == "__main__":
    main()
