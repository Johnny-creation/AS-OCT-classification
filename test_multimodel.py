import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import (
    ResNet34_Weights, ResNet50_Weights, DenseNet169_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, VGG16_Weights,
    ViT_B_16_Weights, ConvNeXt_Tiny_Weights, Swin_T_Weights, MobileNet_V2_Weights,
    ResNeXt50_32X4D_Weights
)
from PIL import Image
import os
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from dataset_utils import ASOCTDatasetJSON


def get_model(model_name, num_classes):
    """根据模型名称获取相应的模型"""
    if model_name == 'resnet34':
        weights = ResNet34_Weights.IMAGENET1K_V1
        model = models.resnet34(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'resnext50':
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        model = models.resnext50_32x4d(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'densenet169':
        weights = DenseNet169_Weights.IMAGENET1K_V1
        model = models.densenet169(weights=weights)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b3':
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1
        model = models.efficientnet_b3(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'efficientnet_b4':
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = models.efficientnet_b4(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vgg16':
        weights = VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vit':
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'convnext_tiny':
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'swin_t':
        weights = Swin_T_Weights.IMAGENET1K_V1
        model = models.swin_t(weights=weights)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        model = models.mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


def test_model(model_name='resnet34'):
    """测试单个模型"""
    print(f"测试模型: {model_name}")

    # 根据模型选择输入大小
    model = get_model(model_name, 4)

    # 定义数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建测试数据集
    test_dataset = ASOCTDatasetJSON('dataset/asoct.test.json', data_transforms)

    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 获取类别信息
    class_names = test_dataset.classes
    num_classes = len(class_names)

    # 使用GPU如果可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载训练好的模型权重
    model_save_path = f'weights/best_{model_name}_model.pth'
    if not os.path.exists(model_save_path):
        print(f"错误: 模型权重文件 {model_save_path} 不存在")
        return None

    model.load_state_dict(torch.load(model_save_path, weights_only=True))
    model = model.to(device)
    model.eval()

    # 存储预测结果和真实标签
    all_preds = []
    all_labels = []

    # 测试过程
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

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

    # 打印评估指标
    print(f"\n{model_name.upper()} 模型评估结果:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 打印分类报告
    print(f"\n{model_name.upper()} 分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 创建results目录保存图表
    results_dir = 'results/evaluation'
    os.makedirs(results_dir, exist_ok=True)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name.upper()} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 计算每个类别的准确率
    class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # 绘制各类别准确率
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracy)
    plt.title(f'{model_name.upper()} Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)

    # 在柱状图上添加数值标签
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{results_dir}/{model_name}_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser(description='测试训练好的模型')
    parser.add_argument('--models', type=str,
                        default='resnet34+resnet50+resnext50+densenet169+efficientnet_b3+efficientnet_b4+vgg16+vit+convnext_tiny+swin_t+mobilenet_v2',
                        help='要测试的模型，多个模型用+分隔')
    args = parser.parse_args()

    # 解析模型名称
    model_names = args.models.split('+')

    # 验证模型名称
    supported_models = ['resnet34', 'resnet50', 'resnext50', 'densenet169', 'efficientnet_b3', 'efficientnet_b4', 'vgg16',
                       'vit', 'convnext_tiny', 'swin_t', 'mobilenet_v2']

    for model_name in model_names:
        if model_name not in supported_models:
            print(f"警告: 不支持的模型 {model_name}")
            continue

    # 测试每个模型
    all_results = []
    for model_name in model_names:
        if model_name in supported_models:
            result = test_model(model_name)
            if result:
                all_results.append(result)

    # 生成所有模型评估结果对比图
    if len(all_results) > 1:
        print(f"\n生成模型对比图...")

        # 创建results目录
        results_dir = 'results/evaluation'
        os.makedirs(results_dir, exist_ok=True)

        # 准备数据
        models = [r['model_name'] for r in all_results]
        accuracies = [r['accuracy'] for r in all_results]
        precisions = [r['precision'] for r in all_results]
        recalls = [r['recall'] for r in all_results]
        f1_scores = [r['f1'] for r in all_results]

        # 绘制模型对比图
        x = np.arange(len(models))
        width = 0.2

        fig, ax = plt.subplots(figsize=(15, 8))
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
                           ha='center', va='bottom', fontsize=8)

        add_value_labels(rects1)
        add_value_labels(rects2)
        add_value_labels(rects3)
        add_value_labels(rects4)

        # 设置图表属性
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)

        # 保存图表和结果摘要
        plt.tight_layout()
        plt.savefig(f'{results_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 保存结果摘要到JSON
        results_summary = {model['model_name']: model for model in all_results}
        with open(f'{results_dir}/evaluation_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"模型对比图已保存到 {results_dir}/model_comparison.png")
        print(f"评估结果已保存到 {results_dir}/evaluation_results.json")

    print(f"\n所有模型测试完成！")


if __name__ == '__main__':
    main()