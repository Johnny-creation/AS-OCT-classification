import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import (
    ResNet50_Weights, DenseNet169_Weights,
    EfficientNet_B4_Weights, VGG16_Weights,
    ConvNeXt_Tiny_Weights, MobileNet_V2_Weights,
    ResNeXt50_32X4D_Weights
)
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import argparse
import json
from advanced_ensemble import ENSEMBLE_METHODS
from dataset_utils import ASOCTDatasetJSON


def get_model(model_name, num_classes):
    """根据模型名称获取相应的模型"""
    if model_name == 'resnet50':
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
    elif model_name == 'convnext_tiny':
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        num_ftrs = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.IMAGENET1K_V2
        model = models.mobilenet_v2(weights=weights)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    return model


def get_target_layers(model, model_name):
    """获取不同模型的目标层"""
    if 'resnet' in model_name or 'resnext' in model_name:
        return [model.layer4[-1]]
    elif 'densenet' in model_name:
        return [model.features[-1]]
    elif 'efficientnet' in model_name:
        return [model.features[-1]]
    elif 'vgg' in model_name:
        return [model.features[-1]]
    elif 'convnext' in model_name:
        return [model.features[-1]]
    elif 'mobilenet' in model_name:
        return [model.features[-1]]
    else:
        return [model.layer4[-1]]  # default


class EnsembleModel(nn.Module):
    """集成模型包装器"""
    def __init__(self, models_dict, ensemble_method, ensemble_model_path=None):
        super(EnsembleModel, self).__init__()
        self.models_dict = models_dict
        self.ensemble_method = ensemble_method
        self.model_names = list(models_dict.keys())
        self.n_classes = 4

        # 如果提供了集成模型路径，加载集成模型
        if ensemble_model_path and os.path.exists(ensemble_model_path):
            self.ensemble_model = ENSEMBLE_METHODS[ensemble_method](n_classes=self.n_classes)
            self.ensemble_model.load(ensemble_model_path)
        else:
            self.ensemble_model = None

    def forward(self, x):
        # 获取所有模型的预测
        all_outputs = []
        for model_name, model in self.models_dict.items():
            output = model(x)
            probs = F.softmax(output, dim=1)
            all_outputs.append(probs)

        # 连接所有输出
        combined_features = torch.cat(all_outputs, dim=1)

        # 如果有训练好的集成模型，使用它；否则使用简单平均
        if self.ensemble_model:
            # 转换为numpy进行集成预测
            combined_np = combined_features.detach().cpu().numpy()
            ensemble_probs = self.ensemble_model.prediction(combined_np)
            return torch.from_numpy(ensemble_probs).to(x.device)
        else:
            # 简单平均集成
            ensemble_output = torch.mean(torch.stack(all_outputs), dim=0)
            return ensemble_output


def load_image_for_heatmap(image_path):
    """加载并预处理图像用于热力图生成"""
    # 加载原始图像
    image = Image.open(image_path).convert('RGB')

    # 转换为numpy数组（用于显示）
    rgb_img = np.array(image) / 255.0

    # 预处理用于模型输入
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0)

    return input_tensor, rgb_img


def generate_individual_model_heatmap(model, model_name, image_path, class_names, device, output_dir):
    """为单个模型生成热力图"""
    print(f"为模型 {model_name} 生成热力图...")

    # 加载图像
    input_tensor, rgb_img = load_image_for_heatmap(image_path)
    input_tensor = input_tensor.to(device)

    # 获取目标层
    target_layers = get_target_layers(model, model_name)

    # 创建GradCAM对象
    cam = GradCAM(model=model, target_layers=target_layers)

    # 获取模型预测
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_class].item()

    # 为每个类别生成热力图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    # 为每个类别生成热力图
    for i, class_name in enumerate(class_names):
        if i >= 4:  # 最多显示4个类别
            break

        targets = [ClassifierOutputTarget(i)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # 调整大小以匹配原图
        resized_rgb = cv2.resize(rgb_img, (224, 224))
        visualization = show_cam_on_image(resized_rgb, grayscale_cam, use_rgb=True)

        axes[i].imshow(visualization)
        class_prob = probs[0, i].item()
        title = f'{class_name}\nProb: {class_prob:.3f}'
        if i == predicted_class:
            title += ' ★'
        axes[i].set_title(title, fontsize=12, fontweight='bold' if i == predicted_class else 'normal')
        axes[i].axis('off')

    plt.suptitle(f'{model_name.upper()} - Predicted: {class_names[predicted_class]} (Conf: {confidence:.3f})')
    plt.tight_layout()

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{model_name}_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"热力图已保存到: {save_path}")
    return predicted_class, confidence


def calculate_ensemble_prediction(models_dict, ensemble_method, image_path, class_names, device, ensemble_model_path=None):
    """计算集成模型预测结果（不生成热力图）"""
    print(f"计算集成预测 ({ensemble_method})...")

    # 加载图像
    input_tensor, _ = load_image_for_heatmap(image_path)
    input_tensor = input_tensor.to(device)

    # 获取各个模型的预测结果
    individual_predictions = {}
    all_outputs = []

    for model_name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            all_outputs.append(probs[0].cpu().numpy())

            individual_predictions[model_name] = {
                'probs': probs[0].cpu().numpy(),
                'predicted': torch.argmax(probs, dim=1).item(),
                'confidence': torch.max(probs, dim=1)[0].item()
            }

    # 简单的平均集成（如果没有训练好的集成模型）
    if ensemble_method == 'MeanWeighted' or not ensemble_model_path:
        # 简单平均
        ensemble_probs = np.mean(all_outputs, axis=0)
        predicted_class = np.argmax(ensemble_probs)
        confidence = ensemble_probs[predicted_class]
    else:
        # 这里可以添加其他集成方法的支持
        ensemble_probs = np.mean(all_outputs, axis=0)
        predicted_class = np.argmax(ensemble_probs)
        confidence = ensemble_probs[predicted_class]

    print(f"集成预测: {class_names[predicted_class]} (置信度: {confidence:.3f})")

    return predicted_class, confidence


def generate_ensemble_heatmap(models_dict, ensemble_method, image_path, class_names, device, output_dir, ensemble_model_path=None):
    """为集成模型生成热力图"""
    print(f"为集成模型 ({ensemble_method}) 生成热力图...")

    # 创建集成模型
    ensemble_model = EnsembleModel(models_dict, ensemble_method, ensemble_model_path)
    ensemble_model = ensemble_model.to(device)
    ensemble_model.eval()

    # 加载图像
    input_tensor, rgb_img = load_image_for_heatmap(image_path)
    input_tensor = input_tensor.to(device)

    # 获取集成预测
    with torch.no_grad():
        ensemble_output = ensemble_model(input_tensor)
        ensemble_probs = F.softmax(ensemble_output, dim=1)
        predicted_class = torch.argmax(ensemble_probs, dim=1).item()
        confidence = ensemble_probs[0, predicted_class].item()

    # 获取各个模型的预测结果
    individual_predictions = {}
    for model_name, model in models_dict.items():
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            individual_predictions[model_name] = {
                'probs': probs[0].cpu().numpy(),
                'predicted': torch.argmax(probs, dim=1).item(),
                'confidence': torch.max(probs, dim=1)[0].item()
            }

    # 创建综合可视化
    n_models = len(models_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.ravel()

    # 显示各个模型的预测结果
    for i, (model_name, pred_info) in enumerate(individual_predictions.items()):
        if i >= len(axes):
            break

        # 创建条形图显示概率分布
        axes[i].bar(range(len(class_names)), pred_info['probs'])
        axes[i].set_xticks(range(len(class_names)))
        axes[i].set_xticklabels(class_names, rotation=45, ha='right')
        axes[i].set_ylim(0, 1)
        axes[i].set_title(f'{model_name}\nPred: {class_names[pred_info["predicted"]]}\nConf: {pred_info["confidence"]:.3f}')

        # 高亮预测类别
        axes[i].bar(pred_info['predicted'], pred_info['probs'][pred_info['predicted']],
                        color='red', alpha=0.7)

    # 隐藏多余的子图
    for i in range(n_models, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Ensemble ({ensemble_method}) - Predicted: {class_names[predicted_class]} (Conf: {confidence:.3f})')
    plt.tight_layout()

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'ensemble_{ensemble_method}_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 生成集成概率分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 个体模型预测分布
    model_names = list(individual_predictions.keys())
    pred_matrix = np.array([individual_predictions[name]['probs'] for name in model_names])

    sns.heatmap(pred_matrix, annot=True, fmt='.3f', cmap='viridis',
                xticklabels=class_names, yticklabels=model_names, ax=ax1)
    ax1.set_title('Individual Model Predictions')

    # 集成预测结果
    ensemble_probs_np = ensemble_probs[0].cpu().numpy()
    ax2.bar(range(len(class_names)), ensemble_probs_np)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_xticklabels(class_names)
    ax2.set_ylim(0, 1)
    ax2.set_title('Ensemble Prediction')
    ax2.bar(predicted_class, ensemble_probs_np[predicted_class], color='red', alpha=0.7)

    plt.tight_layout()
    save_path2 = os.path.join(output_dir, f'ensemble_{ensemble_method}_distribution.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"集成分析图已保存到: {save_path}")
    print(f"集成分布图已保存到: {save_path2}")

    return predicted_class, confidence, individual_predictions


def main():
    parser = argparse.ArgumentParser(description='生成模型热力图可视化')
    parser.add_argument('--image_path', type=str,
                        help='要分析的图像路径（如不指定，将分析sample文件夹中的所有图像）')
    parser.add_argument('--sample_mode', action='store_true', default=True,
                        help='批量分析sample文件夹中的所有图像（默认开启）')
    parser.add_argument('--models', type=str,
                        default='resnet50+densenet169+efficientnet_b4',
                        help='要使用的模型，用+分隔')
    parser.add_argument('--ensemble_method', type=str, default='MeanWeighted',
                        choices=list(ENSEMBLE_METHODS.keys()),
                        help='集成方法')
    parser.add_argument('--output_dir', type=str, default='results/heatmaps/sample_analysis',
                        help='输出目录')
    parser.add_argument('--ensemble_model_path', type=str,
                        help='训练好的集成模型路径（可选）')
    args = parser.parse_args()

    # 确定要分析的图像列表
    image_list = []

    if args.image_path:
        # 如果指定了具体图像路径
        if not os.path.exists(args.image_path):
            print(f"错误: 图像文件 {args.image_path} 不存在")
            return
        image_list = [args.image_path]
        args.sample_mode = False
    elif args.sample_mode:
        # 默认分析sample文件夹中的所有图像
        sample_dir = './sample'
        if not os.path.exists(sample_dir):
            print(f"错误: sample文件夹 {sample_dir} 不存在")
            return

        # 查找sample文件夹中的图像文件
        for filename in os.listdir(sample_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(os.path.join(sample_dir, filename))

        if not image_list:
            print(f"错误: sample文件夹中没有找到图像文件")
            return

        print(f"找到 {len(image_list)} 个sample图像: {[os.path.basename(img) for img in image_list]}")
    else:
        print("错误: 请指定 --image_path 或使用默认的 --sample_mode")
        return

    # 解析模型名称
    model_names = args.models.split('+')

    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 类别名称
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = ASOCTDatasetJSON('dataset/asoct.test.json', data_transforms)
    class_names = test_dataset.classes
    num_classes = len(class_names)

    print(f"类别: {class_names}")
    print(f"集成方法: {args.ensemble_method}")
    print(f"使用模型: {model_names}")

    # 加载所有模型
    models_dict = {}
    print("\n=== 加载模型 ===")
    for model_name in model_names:
        print(f"加载模型: {model_name}")

        # 加载模型
        model = get_model(model_name, num_classes)
        model_save_path = f'weights/best_{model_name}_model.pth'

        if not os.path.exists(model_save_path):
            print(f"警告: 模型权重文件 {model_save_path} 不存在，跳过")
            continue

        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        model = model.to(device)
        model.eval()
        models_dict[model_name] = model

    if not models_dict:
        print("错误: 没有找到可用的模型权重文件")
        return

    print(f"成功加载 {len(models_dict)} 个模型: {list(models_dict.keys())}")

    # 分析每个图像
    all_results = {}

    for i, image_path in enumerate(image_list):
        print(f"\n{'='*60}")
        print(f"分析图像 {i+1}/{len(image_list)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        # 为每个图像创建单独的输出目录
        if args.sample_mode:
            image_name = os.path.splitext(os.path.basename(image_path))[0].lower()
            current_output_dir = os.path.join(args.output_dir, image_name)
        else:
            current_output_dir = args.output_dir

        # 生成个体模型热力图
        individual_results = {}
        print("\n=== 个体模型分析 ===")
        for model_name, model in models_dict.items():
            pred_class, confidence = generate_individual_model_heatmap(
                model, model_name, image_path, class_names, device, current_output_dir
            )

            individual_results[model_name] = {
                'predicted_class': pred_class,
                'predicted_name': class_names[pred_class],
                'confidence': confidence
            }

        print("\n=== 集成模型分析 ===")
        # 计算集成预测（不生成热力图）
        ensemble_pred_class, ensemble_confidence = calculate_ensemble_prediction(
            models_dict, args.ensemble_method, image_path, class_names, device,
            args.ensemble_model_path
        )

        # 生成总结报告
        print(f"\n=== 分析总结 ===")
        print(f"图像: {os.path.basename(image_path)}")
        print(f"集成预测: {class_names[ensemble_pred_class]} (置信度: {ensemble_confidence:.3f})")

        print(f"\n个体模型预测:")
        for model_name, result in individual_results.items():
            print(f"  {model_name}: {result['predicted_name']} (置信度: {result['confidence']:.3f})")

        # 保存结果到JSON
        results_summary = {
            'image_path': image_path,
            'ensemble_method': args.ensemble_method,
            'ensemble_prediction': {
                'class_index': int(ensemble_pred_class),
                'class_name': class_names[ensemble_pred_class],
                'confidence': float(ensemble_confidence)
            },
            'individual_predictions': individual_results,
            'class_names': class_names
        }

        os.makedirs(current_output_dir, exist_ok=True)
        summary_path = os.path.join(current_output_dir, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)

        print(f"\n分析结果已保存到: {summary_path}")
        print(f"个体模型热力图已保存到: {current_output_dir}")

        all_results[os.path.basename(image_path)] = results_summary

    # 如果分析了多个图像，生成汇总报告
    if len(image_list) > 1:
        print(f"\n{'='*60}")
        print(f"汇总报告 - 共分析 {len(image_list)} 个图像")
        print(f"{'='*60}")

        summary_table = []
        for image_name, result in all_results.items():
            ensemble_pred = result['ensemble_prediction']
            individual_preds = result['individual_predictions']

            # 检查个体模型预测一致性
            pred_classes = [pred['predicted_class'] for pred in individual_preds.values()]
            is_consistent = len(set(pred_classes)) == 1

            summary_table.append({
                'image': image_name,
                'ensemble_prediction': ensemble_pred['class_name'],
                'ensemble_confidence': f"{ensemble_pred['confidence']:.3f}",
                'model_consistency': '✅ 一致' if is_consistent else '❌ 不一致',
                'avg_individual_confidence': f"{np.mean([pred['confidence'] for pred in individual_preds.values()]):.3f}"
            })

        print(f"\n{'图像':<20} {'集成预测':<15} {'集成置信度':<10} {'模型一致性':<10} {'平均置信度':<10}")
        print("-" * 70)
        for row in summary_table:
            print(f"{row['image']:<20} {row['ensemble_prediction']:<15} {row['ensemble_confidence']:<10} {row['model_consistency']:<10} {row['avg_individual_confidence']:<10}")

        # 保存汇总结果
        summary_report_path = os.path.join(args.output_dir, 'batch_analysis_summary.json')
        with open(summary_report_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n汇总分析结果已保存到: {summary_report_path}")

    print(f"\n🎉 分析完成！所有结果已保存到: {args.output_dir}")


if __name__ == '__main__':
    main()