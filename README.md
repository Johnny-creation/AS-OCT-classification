# AS-OCT图像分类项目

本项目使用多种深度学习模型对AS-OCT（前节光学相干断层扫描）图像进行分类。

# AS-OCT图像分类项目
## 支持的模型

- ResNet-34
- DenseNet-169
- EfficientNet-B7
- VGG16
- Vision Transformer (ViT)
- Inception v3
- ConvNeXt Tiny
- Swin Transformer Tiny
- MobileNetV2


## 项目结构

```
AS-OCT/
├── data/                      # 原始数据集根目录
│   ├── Cataract/              # 白内障类别，内含多个子文件夹
│   ├── Normal/                # 正常类别
│   ├── PACG/                  # 闭角型青光眼类别
│   ├── PACG_Cataract/         # 闭角型青光眼合并白内障类别
├── dataset/                   # 数据划分文件夹
│   ├── train.txt              # 训练集文件列表
│   ├── val.txt                # 验证集文件列表
│   └── test.txt               # 测试集文件列表
├── figs/                      # 训练/测试生成的图表
│   ├── class_accuracy.png     # 各类别准确率柱状图
│   └── confusion_matrix.png   # 混淆矩阵热力图
├── weights/                   # 保存模型权重
│   └── best_resnet34_model.pth# 示例：最佳模型权重
├── split.py                   # 数据集划分脚本
├── train_multimodel.py        # 多模型训练脚本
├── test_multimodel.py         # 多模型测试脚本
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明文件
```

## 环境配置

1. 安装项目依赖：

```bash
pip install -r requirements.txt
```


## 数据准备

1. **数据集放置**
	 - 请将原始AS-OCT图像数据按类别分别放入 `data` 文件夹下的对应子文件夹（如 `Cataract`、`Normal`、`PACG`、`PACG_Cataract`）。
	 - 每个类别下应包含若干子文件夹（如234OD、234OS等），每个子文件夹内为该类别的所有图像（jpg格式）。
	 - 目录示例：
		 ```
		 data/
			 ├── Cataract/
			 │     ├── 234OD/
			 │     │     ├── 1.jpg
			 │     │     ├── 2.jpg
			 │     │     └── ...
			 │     └── ...
			 ├── Normal/
			 ├── PACG/
			 └── PACG_Cataract/
		 ```

2. **生成数据集划分文件**
	 - 运行 `split.py` 脚本自动划分训练集、验证集和测试集，并在 `dataset/` 文件夹下生成 `train.txt`、`val.txt`、`test.txt`。
	 - 执行命令：
		 ```bash
		 python split.py
		 ```
	 - 每个txt文件每行格式为：`类别/子文件夹/图片名.jpg`，如：`Cataract/234OD/1.jpg`

3. **数据划分比例**
	 - 默认训练集:验证集:测试集 = 7:1.5:1.5（可在 `split.py` 中修改比例）。




## 模型训练

训练脚本支持多种模型选择，并可通过命令行参数灵活调整训练轮数、批量大小等参数。

常用参数说明（`train_multimodel.py`）：

- `--model`：选择要训练的模型（如resnet34、densenet169等）
- `--epochs`：训练轮数（默认5）
- `--batch_size`：批量大小（默认32）
- `--lr`：学习率（默认0.001）

示例命令：
```bash
# 训练ResNet-34模型，10轮，批量64，学习率0.0005
python train_multimodel.py --model resnet34 --epochs 10 --batch_size 64 --lr 0.0005

# 训练DenseNet-169模型，默认参数
python train_multimodel.py --model densenet169
```

训练过程中会显示进度条和损失值，并自动保存验证集上表现最好的模型权重。

## 模型测试

测试脚本同样支持多种模型选择：

```bash
# 测试ResNet-34模型
python test_multimodel.py --model resnet34

# 测试DenseNet-169模型
python test_multimodel.py --model densenet169

# 测试EfficientNet-B7模型
python test_multimodel.py --model efficientnet_b7

# 测试VGG16模型
python test_multimodel.py --model vgg16

# 测试Vision Transformer模型
python test_multimodel.py --model vit

# 测试Inception v3模型
python test_multimodel.py --model inception_v3

# 测试ConvNeXt Tiny模型
python test_multimodel.py --model convnext_tiny

# 测试Swin Transformer Tiny模型
python test_multimodel.py --model swin_t

# 测试MobileNetV2模型
python test_multimodel.py --model mobilenet_v2
```

测试完成后会输出以下评估指标：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1分数

同时会生成并保存：
- 详细分类报告
- 混淆矩阵热力图
- 各类别准确率柱状图

## 输出文件

训练和测试过程中会生成以下文件：
- `weights/best_{model_name}_model.pth`：训练好的模型权重文件（如`best_resnet34_model.pth`）
- `figs/{model_name}_confusion_matrix.png`：混淆矩阵热力图
- `figs/{model_name}_class_accuracy.png`：各类别准确率柱状图