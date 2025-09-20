import os
import random
import json
from collections import defaultdict

# 设置随机种子以确保可复现性
random.seed(42)

# 定义数据集根目录
root_dir = r'./data'

# 定义划分比例 - 参考医疗项目的4分割策略
sampling_splits = [0.65, 0.10, 0.10, 0.15]  # 训练/模型验证/集成验证/测试
sampling_names = ["train-model", "val-model", "val-ensemble", "test"]

# 定义类别
categories = ['Cataract', 'Normal', 'PACG', 'PACG_Cataract']

# 患者级别数据划分：防止数据泄漏
def extract_patient_id(folder_name):
    """从文件夹名称提取患者ID，例如从 '780OS' 提取 '780'"""
    # 去除后缀（如OS/OD）获取患者ID
    if folder_name.endswith('OS') or folder_name.endswith('OD'):
        return folder_name[:-2]
    else:
        # 如果没有OS/OD后缀，直接返回文件夹名
        return folder_name

def patient_level_split(category_data, splits, names):
    """按患者级别进行数据划分"""
    # 按患者ID分组
    patient_groups = defaultdict(list)
    for folder in category_data:
        patient_id = extract_patient_id(folder)
        patient_groups[patient_id].append(folder)

    # 获取所有患者ID并随机打乱
    patient_ids = list(patient_groups.keys())
    random.shuffle(patient_ids)

    # 计算每个划分的患者数量
    n_patients = len(patient_ids)
    split_indices = []
    start_idx = 0

    for i, split_ratio in enumerate(splits):
        if i == len(splits) - 1:  # 最后一个划分取剩余所有患者
            end_idx = n_patients
        else:
            end_idx = start_idx + int(n_patients * split_ratio)
        split_indices.append((start_idx, end_idx))
        start_idx = end_idx

    # 按患者ID划分数据
    splits_result = {}
    for i, (name, (start_idx, end_idx)) in enumerate(zip(names, split_indices)):
        split_patients = patient_ids[start_idx:end_idx]
        split_folders = []
        for patient_id in split_patients:
            split_folders.extend(patient_groups[patient_id])
        splits_result[name] = split_folders
        print(f"  - {name}: {len(split_patients)}个患者, {len(split_folders)}个文件夹")

    return splits_result

# 创建一个字典来存储所有划分结果
all_splits = {}

# 遍历每个类别，进行患者级别划分
for category in categories:
    category_source_dir = os.path.join(root_dir, category)
    if not os.path.exists(category_source_dir):
        print(f"警告: 类别目录 {category_source_dir} 不存在，已跳过。")
        continue

    # 获取该类别下的所有子文件夹
    subfolders = [f.name for f in os.scandir(category_source_dir) if f.is_dir()]

    print(f"类别 {category} 患者级别划分:")
    print(f"  - 总文件夹数: {len(subfolders)}")

    # 进行患者级别划分
    category_splits = patient_level_split(subfolders, sampling_splits, sampling_names)
    all_splits[category] = category_splits

# 获取子文件夹中的jpg图片
def get_jpg_files(category, subfolder):
    subfolder_path = os.path.join(root_dir, category, subfolder)
    jpg_files = [f.name for f in os.scandir(subfolder_path) if f.is_file() and f.name.lower().endswith('.jpg')]
    return jpg_files

# 创建JSON格式的数据集文件
def create_dataset_files(all_splits, sampling_names):
    """创建JSON格式的数据集文件"""
    # 确保dataset目录存在
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    # 为每个划分创建JSON文件
    for split_name in sampling_names:
        json_file = os.path.join('dataset', f'asoct.{split_name}.json')

        # 收集该划分的所有数据
        split_data = []

        for category, splits in all_splits.items():
            if split_name not in splits:
                continue

            for subfolder in splits[split_name]:
                jpg_files = get_jpg_files(category, subfolder)
                for jpg_file in jpg_files:
                    # 构建完整路径
                    rel_path = f"{category}/{subfolder}/{jpg_file}"
                    abs_path = os.path.abspath(os.path.join(root_dir, rel_path))

                    # 添加到JSON数据
                    split_data.append({
                        "path": abs_path,
                        "label": category,
                        "relative_path": rel_path
                    })

        # 写入JSON文件
        json_data = {
            "dataset_name": "asoct",
            "split_name": split_name,
            "total_samples": len(split_data),
            "classes": categories,
            "data": split_data
        }

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"{split_name}: {len(split_data)}个样本")
        print(f"  - JSON文件: {json_file}")

# 生成数据集文件
create_dataset_files(all_splits, sampling_names)

print(f"\n患者级别数据集划分完成！")
print(f"生成文件: asoct.train-model.json, asoct.val-model.json, asoct.val-ensemble.json, asoct.test.json")