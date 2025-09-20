import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os


class ASOCTDatasetJSON(Dataset):
    """从JSON文件加载ASOCT数据集"""

    def __init__(self, json_file, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # 读取JSON文件
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取类别信息
        self.classes = sorted(data['classes'])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 提取样本信息
        for sample in data['data']:
            path = sample['path']
            label = sample['label']
            self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # 加载图像
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[label]
        return image, label_idx


class ASOCTDatasetTXT(Dataset):
    """从TXT文件加载ASOCT数据集（兼容旧代码）"""

    def __init__(self, txt_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    path = line.strip()
                    label = path.split('/')[0]
                    self.samples.append((path, label))

        self.classes = sorted(list(set([sample[1] for sample in self.samples])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        full_path = os.path.join(self.root_dir, path)
        image = Image.open(full_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label_idx = self.class_to_idx[label]
        return image, label_idx


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