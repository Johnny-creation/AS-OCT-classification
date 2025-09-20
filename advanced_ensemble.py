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
import numpy as np
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import pickle
from abc import ABC, abstractmethod
from scipy import stats
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


# 抽象基类 - 参考introduction.md中的设计
class AbstractEnsemble(ABC):
    def __init__(self, n_classes=4):
        self.n_classes = n_classes

    @abstractmethod
    def training(self, train_x, train_y):
        """训练集成模型"""
        pass

    @abstractmethod
    def prediction(self, data):
        """进行预测"""
        pass

    @abstractmethod
    def dump(self, path):
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path):
        """加载模型"""
        pass


# 1. 等权重平均方法
class MeanUnweighted(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)

    def training(self, train_x, train_y):
        pass  # 不需要训练

    def prediction(self, data):
        n_models = data.shape[1] // self.n_classes
        avg_probs = np.zeros((data.shape[0], self.n_classes))
        for i in range(n_models):
            start_idx = i * self.n_classes
            end_idx = (i + 1) * self.n_classes
            avg_probs += data[:, start_idx:end_idx]
        return avg_probs / n_models

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump({}, f)

    def load(self, path):
        pass


# 2. 加权平均方法
class MeanWeighted(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)
        self.weights = None

    def training(self, train_x, train_y):
        # 基于验证集性能计算权重
        n_models = train_x.shape[1] // self.n_classes
        weights = []

        for i in range(n_models):
            start_idx = i * self.n_classes
            end_idx = (i + 1) * self.n_classes
            model_probs = train_x[:, start_idx:end_idx]
            model_preds = np.argmax(model_probs, axis=1)
            accuracy = accuracy_score(train_y, model_preds)
            weights.append(accuracy)

        self.weights = np.array(weights)
        self.weights = self.weights / np.sum(self.weights)  # 归一化

    def prediction(self, data):
        n_models = data.shape[1] // self.n_classes
        weighted_probs = np.zeros((data.shape[0], self.n_classes))

        for i in range(n_models):
            start_idx = i * self.n_classes
            end_idx = (i + 1) * self.n_classes
            weighted_probs += self.weights[i] * data[:, start_idx:end_idx]

        return weighted_probs

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'weights': self.weights}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']


# 3. 硬投票
class MajorityVotingHard(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)

    def training(self, train_x, train_y):
        pass

    def prediction(self, data):
        n_models = data.shape[1] // self.n_classes
        votes = np.zeros((data.shape[0], self.n_classes))

        for i in range(n_models):
            start_idx = i * self.n_classes
            end_idx = (i + 1) * self.n_classes
            model_preds = np.argmax(data[:, start_idx:end_idx], axis=1)

            for j, pred in enumerate(model_preds):
                votes[j, pred] += 1

        return votes / n_models

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump({}, f)

    def load(self, path):
        pass


# 4. 软投票
class MajorityVotingSoft(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)

    def training(self, train_x, train_y):
        pass

    def prediction(self, data):
        n_models = data.shape[1] // self.n_classes
        avg_probs = np.zeros((data.shape[0], self.n_classes))

        for i in range(n_models):
            start_idx = i * self.n_classes
            end_idx = (i + 1) * self.n_classes
            avg_probs += data[:, start_idx:end_idx]

        return avg_probs / n_models

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump({}, f)

    def load(self, path):
        pass


# 5-11. 机器学习方法
class LogisticRegressionEnsemble(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def training(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def prediction(self, data):
        return self.model.predict_proba(data)

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class DecisionTreeEnsemble(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)
        self.model = DecisionTreeClassifier(random_state=42)

    def training(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def prediction(self, data):
        return self.model.predict_proba(data)

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class KNeighborsEnsemble(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)
        self.model = KNeighborsClassifier(n_neighbors=5)

    def training(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def prediction(self, data):
        return self.model.predict_proba(data)

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class SupportVectorMachineEnsemble(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)
        self.model = SVC(probability=True, random_state=42)

    def training(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def prediction(self, data):
        return self.model.predict_proba(data)

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class NaiveBayesEnsemble(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)
        self.model = GaussianNB()

    def training(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def prediction(self, data):
        return self.model.predict_proba(data)

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class GaussianProcessEnsemble(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)
        self.model = GaussianProcessClassifier(random_state=42)

    def training(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def prediction(self, data):
        return self.model.predict_proba(data)

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


# 12. 全局最大置信度方法
class GlobalArgmax(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)

    def training(self, train_x, train_y):
        pass

    def prediction(self, data):
        n_models = data.shape[1] // self.n_classes
        result = np.zeros((data.shape[0], self.n_classes))

        for i in range(data.shape[0]):
            max_confidences = []
            model_probs = []

            for j in range(n_models):
                start_idx = j * self.n_classes
                end_idx = (j + 1) * self.n_classes
                probs = data[i, start_idx:end_idx]
                max_confidences.append(np.max(probs))
                model_probs.append(probs)

            best_model_idx = np.argmax(max_confidences)
            result[i] = model_probs[best_model_idx]

        return result

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump({}, f)

    def load(self, path):
        pass


# 13. 最佳模型方法
class BestModel(AbstractEnsemble):
    def __init__(self, n_classes=4):
        super().__init__(n_classes)
        self.best_model_idx = 0

    def training(self, train_x, train_y):
        n_models = train_x.shape[1] // self.n_classes
        accuracies = []

        for i in range(n_models):
            start_idx = i * self.n_classes
            end_idx = (i + 1) * self.n_classes
            model_probs = train_x[:, start_idx:end_idx]
            model_preds = np.argmax(model_probs, axis=1)
            accuracy = accuracy_score(train_y, model_preds)
            accuracies.append(accuracy)

        self.best_model_idx = np.argmax(accuracies)

    def prediction(self, data):
        start_idx = self.best_model_idx * self.n_classes
        end_idx = (self.best_model_idx + 1) * self.n_classes
        return data[:, start_idx:end_idx]

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'best_model_idx': self.best_model_idx}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.best_model_idx = data['best_model_idx']


# 集成方法字典
ENSEMBLE_METHODS = {
    'MeanUnweighted': MeanUnweighted,
    'MeanWeighted': MeanWeighted,
    'MajorityVoting_Hard': MajorityVotingHard,
    'MajorityVoting_Soft': MajorityVotingSoft,
    'LogisticRegression': LogisticRegressionEnsemble,
    'DecisionTree': DecisionTreeEnsemble,
    'KNeighbors': KNeighborsEnsemble,
    'SupportVectorMachine': SupportVectorMachineEnsemble,
    'NaiveBayes': NaiveBayesEnsemble,
    'GaussianProcess': GaussianProcessEnsemble,
    'GlobalArgmax': GlobalArgmax,
    'BestModel': BestModel
}


def load_model_predictions(model_names, subset_name, results_dir="results"):
    """加载模型预测结果"""
    all_predictions = {}

    for model_name in model_names:
        prediction_file = os.path.join(results_dir, f"predictions_{model_name}_{subset_name}_best.json")

        if not os.path.exists(prediction_file):
            print(f"警告: 预测文件 {prediction_file} 不存在")
            continue

        with open(prediction_file, 'r') as f:
            data = json.load(f)
            predictions = data['predictions']

            # 转换为numpy数组
            pred_array = np.array([predictions[key] for key in sorted(predictions.keys())])
            all_predictions[model_name] = pred_array

    return all_predictions


def prepare_ensemble_data(all_predictions, true_labels):
    """准备集成学习数据"""
    # 将预测结果连接成特征矩阵
    feature_matrix = np.concatenate(list(all_predictions.values()), axis=1)

    return feature_matrix, true_labels


def evaluate_ensemble_method(method_name, method_class, train_x, train_y, test_x, test_y, class_names):
    """评估单个集成方法"""
    print(f"\n评估集成方法: {method_name}")

    # 训练集成模型
    ensemble_model = method_class(n_classes=len(class_names))
    ensemble_model.training(train_x, train_y)

    # 预测
    pred_probs = ensemble_model.prediction(test_x)
    pred_labels = np.argmax(pred_probs, axis=1)

    # 计算指标
    accuracy = accuracy_score(test_y, pred_labels)
    precision = precision_score(test_y, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(test_y, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(test_y, pred_labels, average='weighted', zero_division=0)

    # 计算AUC
    try:
        if len(class_names) > 2:
            y_bin = label_binarize(test_y, classes=range(len(class_names)))
            auc = roc_auc_score(y_bin, pred_probs, average='weighted', multi_class='ovr')
        else:
            auc = roc_auc_score(test_y, pred_probs[:, 1])
    except:
        auc = 0.0

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': pred_labels,
        'probabilities': pred_probs
    }

    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    return results, ensemble_model


def plot_ensemble_results(all_results, class_names, output_dir="figs"):
    """绘制集成学习结果"""
    os.makedirs(output_dir, exist_ok=True)

    # 准备数据
    methods = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']

    # 创建性能对比图
    fig, ax = plt.subplots(figsize=(15, 8))
    x = np.arange(len(methods))
    width = 0.15

    for i, metric in enumerate(metrics):
        values = [all_results[method][metric] for method in methods]
        ax.bar(x + i * width, values, width, label=metric.upper())

    ax.set_xlabel('Ensemble Methods')
    ax.set_ylabel('Score')
    ax.set_title('Ensemble Methods Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制最佳方法的混淆矩阵
    best_method = max(all_results.keys(), key=lambda x: all_results[x]['accuracy'])
    best_predictions = all_results[best_method]['predictions']

    # 这里需要真实标签来绘制混淆矩阵
    print(f"最佳集成方法: {best_method} (准确率: {all_results[best_method]['accuracy']:.4f})")


def main():
    parser = argparse.ArgumentParser(description='高级集成学习系统')
    parser.add_argument('--models', type=str,
                        default='resnet34+resnet50+resnext50+densenet169+efficientnet_b3+efficientnet_b4+vgg16+convnext_tiny+mobilenet_v2',
                        help='参与集成的模型名称，用+分隔')
    parser.add_argument('--ensemble_methods', type=str,
                        default='all',
                        help='要使用的集成方法，用+分隔，或者使用all表示所有方法')
    args = parser.parse_args()

    # 解析模型名称
    model_names = args.models.split('+')

    # 解析集成方法
    if args.ensemble_methods.lower() == 'all':
        ensemble_methods = list(ENSEMBLE_METHODS.keys())
    else:
        ensemble_methods = args.ensemble_methods.split('+')

    print(f"参与集成的模型: {model_names}")
    print(f"使用的集成方法: {ensemble_methods}")

    # 加载数据
    data_dir = r'./data'

    # 数据预处理
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 获取真实标签
    val_ensemble_dataset = ASOCTDatasetJSON('dataset/asoct.val-ensemble.json', data_transforms)
    test_dataset = ASOCTDatasetJSON('dataset/asoct.test.json', data_transforms)
    class_names = test_dataset.classes

    # 加载预测结果
    print("\n加载模型预测结果...")
    val_predictions = load_model_predictions(model_names, 'val-ensemble')
    test_predictions = load_model_predictions(model_names, 'test')

    if len(val_predictions) == 0 or len(test_predictions) == 0:
        print("错误: 未找到足够的预测结果文件。请先运行train_multimodel.py生成预测结果。")
        return

    # 准备集成数据
    val_true_labels = np.array([val_ensemble_dataset.class_to_idx[sample[1]]
                               for sample in val_ensemble_dataset.samples])
    test_true_labels = np.array([test_dataset.class_to_idx[sample[1]]
                                for sample in test_dataset.samples])

    val_features, val_labels = prepare_ensemble_data(val_predictions, val_true_labels)
    test_features, test_labels = prepare_ensemble_data(test_predictions, test_true_labels)

    print(f"集成训练数据形状: {val_features.shape}")
    print(f"集成测试数据形状: {test_features.shape}")

    # 评估所有集成方法
    all_results = {}
    trained_models = {}

    for method_name in ensemble_methods:
        if method_name not in ENSEMBLE_METHODS:
            print(f"警告: 未知的集成方法 {method_name}")
            continue

        method_class = ENSEMBLE_METHODS[method_name]
        results, model = evaluate_ensemble_method(
            method_name, method_class, val_features, val_labels,
            test_features, test_labels, class_names
        )

        all_results[method_name] = results
        trained_models[method_name] = model

        # 保存模型
        os.makedirs("ensemble_models", exist_ok=True)
        model_path = f"ensemble_models/{method_name}.pkl"
        model.dump(model_path)

    # 绘制结果
    plot_ensemble_results(all_results, class_names)

    # 创建results目录结构
    os.makedirs("results/ensemble", exist_ok=True)
    os.makedirs("results/ensemble/models", exist_ok=True)
    os.makedirs("results/ensemble/figures", exist_ok=True)

    # 保存结果到results目录
    results_summary = {}
    for method_name, results in all_results.items():
        results_summary[method_name] = {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'auc': results['auc']
        }

    # 保存结果摘要
    ensemble_results_file = "results/ensemble/ensemble_results.json"
    with open(ensemble_results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    # 移动集成模型到results目录
    if os.path.exists("ensemble_models"):
        import shutil
        if os.path.exists("results/ensemble/models"):
            shutil.rmtree("results/ensemble/models")
        shutil.move("ensemble_models", "results/ensemble/models")

    # 移动图表到results目录
    ensemble_fig_file = "figs/ensemble_comparison.png"
    if os.path.exists(ensemble_fig_file):
        import shutil
        shutil.copy(ensemble_fig_file, "results/ensemble/figures/ensemble_comparison.png")

    print(f"\n集成学习评估完成！")
    print(f"结果已保存到 {ensemble_results_file}")
    print(f"图表已保存到 results/ensemble/figures/ 目录")
    print(f"训练好的集成模型已保存到 results/ensemble/models/ 目录")


if __name__ == '__main__':
    main()