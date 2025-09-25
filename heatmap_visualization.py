"""Generate Grad-CAM heatmaps for AS-OCT images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms

from src.models import available_models, create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise model attention via Grad-CAM")
    parser.add_argument("--image", required=True, help="Path to the AS-OCT image")
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=available_models(),
        help="Backbone to load",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to the fine-tuned weight file (defaults to weights/best_<model>.pth)",
    )
    parser.add_argument("--output", default="reports/figs/gradcam.png", help="Where to save the heatmap")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Optional target class index. If omitted the model's top prediction is used.",
    )
    parser.add_argument("--num-classes", type=int, default=4, help="Number of output classes for the classifier")
    return parser.parse_args()


def _build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _load_image(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB")
    rgb = np.array(image).astype(np.float32) / 255.0
    transform = _build_transform()
    tensor = transform(image).unsqueeze(0)
    return tensor, rgb


def _target_layers(model: torch.nn.Module, model_name: str):
    mapping: Dict[str, Callable[[torch.nn.Module], torch.nn.Module]] = {
        "resnet": lambda m: m.layer4[-1],
        "resnext": lambda m: m.layer4[-1],
        "densenet": lambda m: m.features[-1],
        "efficientnet": lambda m: m.features[-1],
        "vgg": lambda m: m.features[-1],
        "convnext": lambda m: m.features[-1],
        "mobilenet": lambda m: m.features[-1],
    }
    for key, resolver in mapping.items():
        if key in model_name:
            return [resolver(model)]
    return [model.layer4[-1]]


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    weight_path = Path(args.weights) if args.weights else Path("weights") / f"best_{args.model}.pth"

    device = torch.device(args.device)
    model = create_model(args.model, num_classes=args.num_classes, pretrained=False)
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    tensor, rgb = _load_image(image_path)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)
        target_class = int(torch.argmax(probs, dim=1)) if args.target_class is None else args.target_class

    cam = GradCAM(model=model, target_layers=_target_layers(model, args.model))
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    heatmap = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(heatmap).save(output_path)
    print(f"Saved Grad-CAM visualisation to {output_path}")


if __name__ == "__main__":
    main()
