"""Dataset helpers for AS-OCT experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class SampleMetadata:
    """Metadata describing a single AS-OCT sample."""

    path: str
    label: str
    patient_id: str
    image_id: str


def _infer_patient_id(path: str) -> str:
    """Infer patient identifier from a file path when not provided."""

    candidate = Path(path).parent.name or Path(path).stem
    candidate = candidate.replace("_OS", "").replace("_OD", "")
    candidate = candidate.replace("-OS", "").replace("-OD", "")
    if not candidate:
        candidate = Path(path).stem
    return candidate


class ASOCTDatasetJSON(Dataset):
    """Dataset backed by a JSON manifest exported by :mod:`split.py`."""

    def __init__(self, json_file: str, transform: Optional[Any] = None):
        self.transform = transform
        self.samples: List[Tuple[str, int, SampleMetadata]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.classes = sorted(data["classes"])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        for sample in data["data"]:
            path = sample["path"]
            label = sample["label"]
            patient_id = sample.get("patient_id") or _infer_patient_id(path)
            image_id = sample.get("image_id") or Path(path).stem
            meta = SampleMetadata(
                path=path,
                label=label,
                patient_id=str(patient_id),
                image_id=str(image_id),
            )
            self.samples.append((path, self.class_to_idx[label], meta))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        path, label_idx, meta = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx, meta

    def get_metadata(self) -> List[SampleMetadata]:
        """Return the metadata for every sample in dataset order."""

        return [meta for _, _, meta in self.samples]


class ASOCTDatasetTXT(Dataset):
    """从TXT文件加载ASOCT数据集（兼容旧代码）。"""

    def __init__(self, txt_file: str, root_dir: str, transform: Optional[Any] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int, SampleMetadata]] = []

        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                path = line.strip()
                label = path.split("/")[0]
                self.samples.append((path, label))

        classes = sorted({sample[1] for sample in self.samples})
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        self.classes = classes

        updated_samples: List[Tuple[str, int, SampleMetadata]] = []
        for rel_path, label in self.samples:
            label_idx = self.class_to_idx[label]
            full_path = os.path.join(self.root_dir, rel_path)
            meta = SampleMetadata(
                path=full_path,
                label=label,
                patient_id=_infer_patient_id(full_path),
                image_id=Path(full_path).stem,
            )
            updated_samples.append((full_path, label_idx, meta))
        self.samples = updated_samples

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        path, label_idx, meta = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label_idx, meta


def load_dataset(dataset_path, transform=None, root_dir=None):
    """
    智能加载数据集，自动检测JSON或TXT格式

    Args:
        dataset_path: 数据集文件路径
        transform: 数据变换
        root_dir: 数据根目录（仅TXT格式需要）

    Returns:
        Dataset对象
    """
    if dataset_path.endswith('.json'):
        return ASOCTDatasetJSON(dataset_path, transform)
    elif dataset_path.endswith('.txt'):
        if root_dir is None:
            root_dir = './data'
        return ASOCTDatasetTXT(dataset_path, root_dir, transform)
    else:
        raise ValueError(f"不支持的数据集格式: {dataset_path}")


def get_dataset_info(json_file):
    """获取数据集信息"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return {
        'dataset_name': data['dataset_name'],
        'split_name': data['split_name'],
        'total_samples': data['total_samples'],
        'classes': data['classes'],
        'num_classes': len(data['classes'])
    }