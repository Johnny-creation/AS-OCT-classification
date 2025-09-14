import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import (
    ResNet34_Weights, DenseNet169_Weights, 
    EfficientNet_B3_Weights, VGG16_Weights, 
    ViT_B_16_Weights, ConvNeXt_Tiny_Weights, Swin_T_Weights, MobileNet_V2_Weights
)
from PIL import Image
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

# # 设置中文字体支持
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

class ASOCTDataset(Dataset):
    def __init__(self, txt_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # 读取txt文件
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    path = line.strip()
                    # 从路径中提取类别
                    label = path.split('/')[0]
                    self.samples.append((path, label))
        
        # 创建类别到索引的映射
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

def get_model(model_name, num_classes):
    """根据模型名称获取相应的模型"""
    if model_name == 'resnet34':
        weights = ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'densenet169':
        weights = DenseNet169_Weights.IMAGENET1K_V1
        model = models.densenet169(weights=weights)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'efficientnet_b3':
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'vgg16':
        weights = VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'vit':
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'convnext_tiny':
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        # For ConvNeXt, the classifier is called 'classifier' and is a Sequential module
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'swin_t':
        weights = Swin_T_Weights.IMAGENET1K_V1
        model = models.swin_t(weights=weights)
        # For Swin Transformer, the classifier is called 'head'
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        model = models.mobilenet_v2(weights=weights)
        # For MobileNetV2, the classifier is called 'classifier' and is a Sequential module
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        raise ValueError(f"不支持的模型: {model_name}")
    
    return model, input_size

def test_model(model_name='resnet34'):
    # 根据模型选择输入大小
    model, input_size = get_model(model_name, 4)  # 假设有4个类别
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 数据集根目录
    data_dir = r'./data'
    
    # 创建测试数据集
    test_dataset = ASOCTDataset('./dataset/test.txt', data_dir, data_transforms)
    
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 获取类别信息
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    # 使用GPU如果可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 加载训练好的模型权重
    model_save_path = f'weights/best_{model_name}_model.pth'
    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 存储预测结果和真实标签
    all_preds = []
    all_labels = []
    
    # 测试过程
    with torch.no_grad():
        print(f"开始测试模型: {model_name}")
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # 保存结果
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算评价指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"\n模型评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 创建figs目录保存图表
    figs_dir = 'figs'
    os.makedirs(figs_dir, exist_ok=True)

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name.upper()}confusion matrix')
    plt.xlabel('predict label')
    plt.ylabel('true label')
    plt.tight_layout()
    plt.savefig(f'figs/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    # 计算每个类别的准确率
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # 绘制各类别准确率
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracy)
    plt.title(f'{model_name.upper()}accuracy')
    plt.xlabel('class')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    
    # 在柱状图上添加数值标签
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'figs/{model_name}_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    
    return model, accuracy, precision, recall, f1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试不同模型')
    parser.add_argument('--model', type=str, default= 'resnet34+densenet169+efficientnet_b3+vgg16+vit+convnext_tiny+swin_t+mobilenet_v2', 
                        help='选择要测试的模型，多个模型用+分隔，例如: resnet34+densenet169+vgg16')
    args = parser.parse_args()
    
    # 解析模型名称
    models_to_test = args.model.split('+')
    
    # 验证模型名称
    supported_models = ['resnet34', 'densenet169', 'efficientnet_b3', 'vgg16', 'vit', 'convnext_tiny', 'swin_t', 'mobilenet_v2']
    for model_name in models_to_test:
        if model_name not in supported_models:
            raise ValueError(f"不支持的模型: {model_name}")
    
    # 存储所有模型的评估结果
    results = {}
    
    # 测试每个模型
    for model_name in models_to_test:
        model, acc, prec, rec, f1 = test_model(model_name)
        results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
    
    # 生成所有模型评估结果对比图
    # 创建figs目录保存图表
    figs_dir = 'figs'
    os.makedirs(figs_dir, exist_ok=True)
    
    # 准备数据
    model_names = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_names]
    precisions = [results[model]['precision'] for model in model_names]
    recalls = [results[model]['recall'] for model in model_names]
    f1_scores = [results[model]['f1'] for model in model_names]
    
    # 绘制模型对比图
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy')
    rects2 = ax.bar(x - 0.5*width, precisions, width, label='Precision')
    rects3 = ax.bar(x + 0.5*width, recalls, width, label='Recall')
    rects4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1 Score')
    
    # 添加数值标签
    def add_value_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_value_labels(rects1)
    add_value_labels(rects2)
    add_value_labels(rects3)
    add_value_labels(rects4)
    
    # 设置图表属性
    ax.set_xlabel('Models')
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(f'figs/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存