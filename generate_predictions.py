import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    ResNet34_Weights, ResNet50_Weights, DenseNet169_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, VGG16_Weights,
    ViT_B_16_Weights, ConvNeXt_Tiny_Weights, Swin_T_Weights, MobileNet_V2_Weights,
    ResNeXt50_32X4D_Weights
)
from torchvision import models
import os
import json
import numpy as np
from tqdm import tqdm
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


def generate_predictions(model, dataloader, model_name, subset_name, device, class_names):
    """生成模型预测结果并保存为JSON格式"""
    model.eval()
    predictions = {}

    with torch.no_grad():
        sample_idx = 0
        for inputs, labels in tqdm(dataloader, desc=f"生成{model_name}预测结果"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            # 保存每个样本的预测概率
            for i in range(probs.size(0)):
                sample_id = f"sample_{sample_idx}"
                predictions[sample_id] = probs[i].cpu().numpy().tolist()
                sample_idx += 1

    # 创建预测结果字典
    result = {
        "predictions": predictions,
        "legend": class_names,
        "model": model_name,
        "subset": subset_name
    }

    # 创建results目录
    os.makedirs("results", exist_ok=True)

    # 保存预测结果
    output_file = f"results/predictions_{model_name}_{subset_name}_best.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"预测结果已保存到: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='使用训练好的模型生成预测结果')
    parser.add_argument('--models', type=str,
                        default='resnet34+resnet50+resnext50+densenet169+efficientnet_b3+efficientnet_b4+vgg16+vit+convnext_tiny+swin_t+mobilenet_v2',
                        help='要生成预测的模型，用+分隔')
    parser.add_argument('--subsets', type=str,
                        default='val-ensemble+test',
                        help='要生成预测的数据集，用+分隔')
    args = parser.parse_args()

    # 解析模型名称
    model_names = args.models.split('+')
    subset_names = args.subsets.split('+')

    # 支持的模型
    supported_models = ['resnet34', 'resnet50', 'resnext50', 'densenet169', 'efficientnet_b3', 'efficientnet_b4', 'vgg16',
                       'vit', 'convnext_tiny', 'swin_t', 'mobilenet_v2']

    # 检查模型权重文件
    available_models = []
    for model_name in model_names:
        if model_name not in supported_models:
            print(f"跳过不支持的模型: {model_name}")
            continue

        weight_file = f'weights/best_{model_name}_model.pth'
        if not os.path.exists(weight_file):
            print(f"跳过缺少权重文件的模型: {model_name} ({weight_file})")
            continue

        available_models.append(model_name)

    if not available_models:
        print("错误: 没有找到可用的模型权重文件")
        return

    print(f"可用模型: {available_models}")

    # 使用GPU如果可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 为每个数据集和模型生成预测
    for subset_name in subset_names:
        json_file = f'dataset/asoct.{subset_name}.json'
        if not os.path.exists(json_file):
            print(f"跳过不存在的数据集: {json_file}")
            continue

        print(f"\n处理数据集: {subset_name}")

        # 创建数据集
        dataset = ASOCTDatasetJSON(json_file, data_transforms)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        class_names = dataset.classes

        print(f"数据集大小: {len(dataset)}")
        print(f"类别: {class_names}")

        # 为每个可用模型生成预测
        for model_name in available_models:
            print(f"\n生成 {model_name} 在 {subset_name} 上的预测...")

            # 加载模型
            model = get_model(model_name, len(class_names))
            weight_file = f'weights/best_{model_name}_model.pth'

            try:
                model.load_state_dict(torch.load(weight_file, weights_only=True))
                model = model.to(device)

                # 生成预测结果
                generate_predictions(model, dataloader, model_name, subset_name, device, class_names)

            except Exception as e:
                print(f"加载模型 {model_name} 失败: {e}")

    print(f"\n预测结果生成完成！现在可以运行集成学习了。")


if __name__ == '__main__':
    main()