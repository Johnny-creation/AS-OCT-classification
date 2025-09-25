"""Training entry-point for multiple AS-OCT backbones."""

from __future__ import annotations

import argparse
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset_utils import ASOCTDatasetJSON
from src.models import available_models, create_model


LOGGER = logging.getLogger("train_multimodel")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one or more classification backbones on AS-OCT datasets",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["resnet50", "densenet169", "efficientnet_b4"],
        help="Model identifiers to train. Available: %(choices)s" % {"choices": ", ".join(available_models())},
    )
    parser.add_argument(
        "--train-json",
        default="dataset/asoct.train-model.json",
        help="Path to the training JSON manifest",
    )
    parser.add_argument(
        "--val-json",
        default="dataset/asoct.val-model.json",
        help="Path to the validation JSON manifest",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (validation epochs)")
    parser.add_argument("--min-delta", type=float, default=1e-3, help="Minimum improvement to reset patience")
    parser.add_argument(
        "--weights-dir",
        default="weights",
        help="Directory where fine-tuned weights are stored",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not initialise backbones with ImageNet weights",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device string passed to torch.device",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
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


def _unpack_batch(batch: Tuple[torch.Tensor, torch.Tensor, object]) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        inputs, labels, _ = batch
    elif len(batch) == 2:
        inputs, labels = batch
    else:
        raise ValueError("Unexpected batch format: %r" % (batch,))
    return inputs, labels


def _train_single_model(
    model_name: str,
    dataloaders: Dict[str, DataLoader],
    dataset_sizes: Dict[str, int],
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    min_delta: float,
    pretrained: bool,
) -> nn.Module:
    num_classes = len(dataloaders["train"].dataset.classes)  # type: ignore[arg-type]
    model = create_model(model_name, num_classes=num_classes, pretrained=pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_state = deepcopy(model.state_dict())
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        LOGGER.info("[%s] Epoch %d/%d", model_name, epoch + 1, epochs)
        epoch_start = time.time()
        for phase in ("train", "val"):
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch in dataloaders[phase]:
                inputs, labels = _unpack_batch(batch)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            LOGGER.info(
                "[%s] %s loss %.4f acc %.4f (elapsed %.1fs)",
                model_name,
                phase,
                epoch_loss,
                epoch_acc,
                time.time() - epoch_start,
            )

            if phase == "val":
                if epoch_acc > best_acc + min_delta:
                    best_acc = epoch_acc
                    best_state = deepcopy(model.state_dict())
                    patience_counter = 0
                    LOGGER.info("[%s] New best validation accuracy: %.4f", model_name, best_acc)
                else:
                    patience_counter += 1
                    LOGGER.debug(
                        "[%s] Validation improvement below threshold for %d/%d epochs",
                        model_name,
                        patience_counter,
                        patience,
                    )

        if patience_counter >= patience:
            LOGGER.info("[%s] Early stopping triggered after %d epochs", model_name, epoch + 1)
            break

    model.load_state_dict(best_state)
    return model


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    transform = _build_transforms()
    train_dataset = ASOCTDatasetJSON(args.train_json, transform)
    val_dataset = ASOCTDatasetJSON(args.val_json, transform)

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }

    dataset_sizes = {phase: len(ds) for phase, ds in {"train": train_dataset, "val": val_dataset}.items()}

    weights_dir = Path(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    LOGGER.info("Using device %s", device)

    for model_name in args.models:
        if model_name not in available_models():
            raise ValueError(f"Unsupported model: {model_name}")

        LOGGER.info("Training model %s", model_name)
        model = _train_single_model(
            model_name=model_name,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            min_delta=args.min_delta,
            pretrained=not args.no_pretrained,
        )

        weight_path = weights_dir / f"best_{model_name}.pth"
        torch.save(model.state_dict(), weight_path)
        LOGGER.info("Saved weights for %s to %s", model_name, weight_path)


if __name__ == "__main__":
    main()
