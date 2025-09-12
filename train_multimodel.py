import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import (
    ResNet34_Weights, DenseNet169_Weights, 
    EfficientNet_B7_Weights, VGG16_Weights, 
    ViT_B_16_Weights, Inception_V3_Weights,
    ConvNeXt_Tiny_Weights, Swin_T_Weights, MobileNet_V2_Weights
)
from PIL import Image
import os
from tqdm import tqdm
import argparse

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
    elif model_name == 'efficientnet_b7':
        weights = EfficientNet_B7_Weights.IMAGENET1K_V1
        model = models.efficientnet_b7(weights=weights)
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
    elif model_name == 'inception_v3':
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = models.inception_v3(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        # Inception v3 expects input size of 299
        input_size = 299
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

def train_model(model_name='resnet34', batch_size=32, epochs=5, learning_rate=0.001, num_workers=4):
    # 根据模型选择输入大小
    model, input_size = get_model(model_name, 4)  # 假设有4个类别
    
    # 定义数据预处理
    if model_name == 'inception_v3':
        # Inception v3需要299x299的输入
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
    else:
        # 其他模型使用224x224的输入
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
    
    # 数据集根目录
    data_dir = r'.'
    
    # 创建数据集
    image_datasets = {
        'train': ASOCTDataset('dataset/train.txt', data_dir, data_transforms['train']),
        'val': ASOCTDataset('dataset/val.txt', data_dir, data_transforms['val'])
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
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 设置模型为训练模式
            else:
                model.eval()   # 设置模型为评估模式
            
            running_loss = 0.0
            running_corrects = 0
            
            # 使用tqdm显示进度条
            dataloader = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch+1}/{num_epochs}')
            
            # 遍历数据
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 清零参数梯度
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    if model_name == 'inception_v3' and phase == 'train':
                        # Inception v3在训练时返回两个输出
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
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
                
                # 实时更新进度条上的损失值
                dataloader.set_postfix(loss=loss.item())
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 深度复制模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                # 保存最佳模型到weights文件夹
                model_save_path = f'weights/best_{model_name}_model.pth'
                torch.save(best_model_wts, model_save_path)
                print(f'新的最佳模型已保存，准确率: {best_acc:.4f}')
        
        print()
    
    print(f'训练完成。最佳验证准确率: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练不同模型')
    parser.add_argument('--model', type=str, default='resnet34', 
                        choices=['resnet34', 'densenet169', 'efficientnet_b7', 'vgg16', 'vit', 
                                'inception_v3', 'convnext_tiny', 'swin_t', 'mobilenet_v2'],
                        help='选择要训练的模型')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='训练的批量大小')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载的线程数')
    args = parser.parse_args()
    
    model = train_model(
        model_name=args.model,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        num_workers=args.workers
    )
    print(f'模型训练完成并已保存最佳权重到 weights/best_{args.model}_model.pth')