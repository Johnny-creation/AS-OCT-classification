import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import (
    ResNet34_Weights, ResNet50_Weights, DenseNet169_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, VGG16_Weights,
    ConvNeXt_Tiny_Weights, MobileNet_V2_Weights,
    ResNeXt50_32X4D_Weights
)
from PIL import Image
import os
import argparse
import time
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

def train_model(model_names, batch_size=32, epochs=5, learning_rate=0.001, num_workers=4, 
                patience=7, min_delta=0.001):
    # 数据集根目录
    data_dir = r'./data'
    
    # 创建weights文件夹
    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)
    
    # 使用224x224的输入
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 创建数据集
    image_datasets = {
        'train': ASOCTDatasetJSON('dataset/asoct.train-model.json', data_transforms['train']),
        'val': ASOCTDatasetJSON('dataset/asoct.val-model.json', data_transforms['val'])
    }
    
    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    
    print(f"类别数: {num_classes}")
    print(f"类别名称: {class_names}")
    print(f"训练集大小: {dataset_sizes['train']}")
    print(f"验证集大小: {dataset_sizes['val']}")
    
    # 使用GPU如果可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 为每个模型训练
    trained_models = {}
    for model_name in model_names:
        print(f"\n开始训练模型: {model_name}")
        
        # 获取模型
        model = get_model(model_name, num_classes)
        
        # 将模型移动到设备
        model = model.to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 训练参数
        num_epochs = epochs
        best_acc = 0.0
        best_model_wts = None
        
        # 早停机制参数
        patience_counter = 0
        best_val_acc = 0.0
        
        # 训练循环
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            
            # 每个epoch都有训练和验证阶段
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # 设置模型为训练模式
                else:
                    model.eval()   # 设置模型为评估模式
                
                running_loss = 0.0
                running_corrects = 0
                
                # 直接使用数据加载器
                dataloader = dataloaders[phase]

                # 遍历数据
                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # 清零参数梯度
                    optimizer.zero_grad()
                    
                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        
                        # 反向传播和优化（仅在训练阶段）
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                
                # 深度复制模型
                if phase == 'val':
                    # 早停机制检查
                    if epoch_acc > best_val_acc + min_delta:
                        best_val_acc = epoch_acc
                        best_model_wts = model.state_dict()
                        patience_counter = 0  # 重置耐心计数器
                        
                        # 保存最佳模型到weights文件夹
                        model_save_path = f'weights/best_{model_name}_model.pth'
                        torch.save(best_model_wts, model_save_path)
                        print(f'新的最佳模型已保存至 {model_save_path}，准确率: {best_val_acc:.4f}')
                    else:
                        patience_counter += 1
                        print(f'验证准确率未提升，耐心计数器: {patience_counter}/{patience}')
                        
                        # 检查是否达到耐心限制
                        if patience_counter >= patience:
                            print(f'早停机制触发，停止训练 {model_name} 模型')
                            # 加载最佳模型权重
                            if best_model_wts is not None:
                                model.load_state_dict(best_model_wts)
                            break  # 跳出epoch循环
            
            # 如果触发了早停，也跳出epoch循环
            if patience_counter >= patience:
                break
            
            print()
        
        print(f'模型 {model_name} 训练完成。最佳验证准确率: {best_val_acc:.4f}')
        
        # 加载最佳模型权重
        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
        trained_models[model_name] = model
    
    return trained_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练不同模型')
    parser.add_argument('--model', type=str, default=['resnet34', 'resnet50', 'resnext50', 'densenet169', 'efficientnet_b3', 'efficientnet_b4', 'vgg16', 'convnext_tiny', 'mobilenet_v2'],
                        nargs='+',  # 允许接收一个或多个值
                        choices=['resnet34', 'resnet50', 'resnext50', 'densenet169', 'efficientnet_b3', 'efficientnet_b4', 'vgg16', 'convnext_tiny', 'mobilenet_v2'],
                        help='选择要训练的模型（可多选）')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='训练的批量大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载的线程数')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停机制的耐心轮数')
    parser.add_argument('--min_delta', type=float, default=0.0005,
                        help='早停机制的最小改善阈值')
    args = parser.parse_args()
    
    models = train_model(
        model_names=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.workers,
        patience=args.patience,
        min_delta=args.min_delta
    )

    print(f'所有模型训练完成并已保存最佳权重')
    print("要生成预测结果，请运行: python generate_predictions.py")