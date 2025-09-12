import os
import random

# 设置随机种子以确保可复现性
random.seed(42)

# 定义数据集根目录
root_dir = r'.\data'

# 定义划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 定义类别
categories = ['Cataract', 'Normal', 'PACG', 'PACG_Cataract']

# 创建一个字典来存储所有划分结果
all_splits = {}

# 遍历每个类别
for category in categories:
    category_source_dir = os.path.join(root_dir, category)
    if not os.path.exists(category_source_dir):
        print(f"警告: 类别目录 {category_source_dir} 不存在，已跳过。")
        continue
    
    # 获取该类别下的所有子文件夹
    subfolders = [f.name for f in os.scandir(category_source_dir) if f.is_dir()]
    # 随机打乱子文件夹列表
    random.shuffle(subfolders)
    
    # 计算划分索引
    train_end = int(len(subfolders) * train_ratio)
    val_end = int(len(subfolders) * (train_ratio + val_ratio))
    
    # 划分数据集
    train_subfolders = subfolders[:train_end]
    val_subfolders = subfolders[train_end:val_end]
    test_subfolders = subfolders[val_end:]
    
    # 将划分结果存储在字典中
    all_splits[category] = {
        'train': train_subfolders,
        'val': val_subfolders,
        'test': test_subfolders
    }
    
    print(f"类别 {category} 划分完成。")
    print(f"  - 训练集子文件夹数: {len(train_subfolders)}")
    print(f"  - 验证集子文件夹数: {len(val_subfolders)}")
    print(f"  - 测试集子文件夹数: {len(test_subfolders)}")

# 获取子文件夹中的jpg图片
def get_jpg_files(category, subfolder):
    subfolder_path = os.path.join(root_dir, category, subfolder)
    jpg_files = [f.name for f in os.scandir(subfolder_path) if f.is_file() and f.name.lower().endswith('.jpg')]
    return jpg_files

# 生成三个独立的txt文件
train_file = os.path.join('dataset', 'train.txt')
val_file = os.path.join('dataset', 'val.txt')
test_file = os.path.join('dataset', 'test.txt')

with open(train_file, 'w', encoding='utf-8') as f:
    for category, splits in all_splits.items():
        for subfolder in splits['train']:
            jpg_files = get_jpg_files(category, subfolder)
            for jpg_file in jpg_files:
                f.write(f"{category}/{subfolder}/{jpg_file}\n")

with open(val_file, 'w', encoding='utf-8') as f:
    for category, splits in all_splits.items():
        for subfolder in splits['val']:
            jpg_files = get_jpg_files(category, subfolder)
            for jpg_file in jpg_files:
                f.write(f"{category}/{subfolder}/{jpg_file}\n")

with open(test_file, 'w', encoding='utf-8') as f:
    for category, splits in all_splits.items():
        for subfolder in splits['test']:
            jpg_files = get_jpg_files(category, subfolder)
            for jpg_file in jpg_files:
                f.write(f"{category}/{subfolder}/{jpg_file}\n")

print(f"\n所有类别数据集划分完成！")
print(f"训练集结果已保存到 {train_file}")
print(f"验证集结果已保存到 {val_file}")
print(f"测试集结果已保存到 {test_file}")