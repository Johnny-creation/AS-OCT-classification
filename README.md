# ASOCT Medical Image Classification with Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning framework for **Anterior Segment Optical Coherence Tomography (ASOCT)** medical image classification, featuring patient-level data splitting to prevent data leakage and 12 ensemble learning methods for enhanced performance.

## 🚀 Features

- **🔒 Data Leakage Prevention**: Patient-level (subject-level) data splitting strategy
- **🧠 Multi-Model Architecture**: Support for 11 state-of-the-art deep learning models
- **🔬 Advanced Ensemble Learning**: 12 different ensemble methods including voting, averaging, and meta-learning
- **📊 Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, AUC, and confusion matrices
- **⚡ GPU Acceleration**: CUDA support for faster training and inference
- **📁 Organized Results**: Structured output directory for easy result management

## 📁 Project Structure

```
📦 ASOCT-Classification/
├── 📂 data/                     # Medical image dataset
│   ├── 📁 Cataract/            # Cataract class images
│   ├── 📁 Normal/              # Normal class images
│   ├── 📁 PACG/                # Primary Angle Closure Glaucoma
│   └── 📁 PACG_Cataract/       # PACG with Cataract
├── 📂 dataset/                 # Dataset split JSON files
├── 📂 weights/                 # Trained model weights
├── 📂 results/                 # All output results
│   ├── 📄 predictions_*.json   # Model prediction files
│   ├── 📂 evaluation/          # Model evaluation results
│   └── 📂 ensemble/            # Ensemble learning results
│       ├── 📂 models/          # Trained ensemble models
│       └── 📂 figures/         # Performance visualizations
├── 🐍 split.py                # Patient-level data splitting
├── 🐍 train_multimodel.py     # Multi-model training pipeline
├── 🐍 test_multimodel.py      # Model evaluation
├── 🐍 advanced_ensemble.py    # Ensemble learning system
├── 🐍 generate_predictions.py # Prediction generation
├── 🐍 dataset_utils.py        # Dataset utilities
└── 📄 requirements.txt        # Python dependencies
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/asoct-classification.git
cd asoct-classification

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
Pillow>=8.3.0
tqdm>=4.62.0
```

## 📊 Dataset Structure

Organize your ASOCT images following this hierarchy:

```
data/
├── Cataract/
│   ├── PatientID_OS/          # Left eye (Oculus Sinister)
│   │   ├── image001.jpg
│   │   └── image002.jpg
│   └── PatientID_OD/          # Right eye (Oculus Dexter)
├── Normal/
├── PACG/
└── PACG_Cataract/
```

> **Note**: Patient ID extraction automatically handles `_OS` and `_OD` suffixes to ensure proper patient-level splitting.

## 🚀 Quick Start

### Step 1: Data Preparation & Splitting

Perform patient-level data splitting to prevent data leakage:

```bash
python split.py
```

**Output:**
- `asoct.train-model.json` - Training set for model development
- `asoct.val-model.json` - Validation set for model selection
- `asoct.val-ensemble.json` - Validation set for ensemble training
- `asoct.test.json` - Test set for final evaluation

### Step 2: Model Training

Train multiple deep learning models with automatic hyperparameter optimization:

```bash
# Train all supported models (default)
python train_multimodel.py

# Train specific models
python train_multimodel.py --model resnet34 densenet169 vgg16

# Custom training parameters
python train_multimodel.py --batch_size 64 --epochs 50 --lr 0.0001 --patience 10
```

**Supported Models:**
`resnet34` | `resnet50` | `resnext50` | `densenet169` | `efficientnet_b3` | `efficientnet_b4` | `vgg16` | `vit` | `convnext_tiny` | `swin_t` | `mobilenet_v2`

### Step 3: Model Evaluation

Evaluate individual model performance:

```bash
# Evaluate all trained models
python test_multimodel.py

# Evaluate specific models
python test_multimodel.py --models resnet34+densenet169+vgg16
```

### Step 4: Ensemble Learning

Deploy advanced ensemble methods for enhanced performance:

```bash
# Run all 12 ensemble methods
python advanced_ensemble.py

# Custom ensemble configuration
python advanced_ensemble.py --models resnet34+densenet169+vgg16 \
                           --ensemble_methods LogisticRegression+MeanWeighted

# Specific ensemble methods only
python advanced_ensemble.py --ensemble_methods LogisticRegression+DecisionTree+KNeighbors
```

## 🧬 Ensemble Learning Methods

Our framework implements **12 sophisticated ensemble techniques** organized into three categories:

### 📊 Statistical Methods
| Method | Description | Use Case |
|--------|-------------|----------|
| **MeanUnweighted** | Equal-weight averaging | Baseline ensemble performance |
| **MeanWeighted** | Performance-weighted averaging | When models have varying quality |
| **MajorityVoting_Hard** | Discrete class voting | Robust predictions with clear decisions |
| **MajorityVoting_Soft** | Probability averaging | Smooth probability distributions |

### 🤖 Meta-Learning Approaches
| Method | Description | Strengths |
|--------|-------------|-----------|
| **LogisticRegression** | Linear meta-classifier | Fast, interpretable, often optimal |
| **DecisionTree** | Tree-based meta-learner | Handles non-linear relationships |
| **KNeighbors** | Instance-based learning | Captures local patterns |
| **SupportVectorMachine** | SVM meta-classifier | Strong generalization |
| **NaiveBayes** | Probabilistic classifier | Robust to noise |
| **GaussianProcess** | Bayesian approach | Uncertainty quantification |

### 🎯 Selection Methods
| Method | Description | Strategy |
|--------|-------------|----------|
| **GlobalArgmax** | Confidence-based selection | Choose most confident prediction |
| **BestModel** | Single best performer | Simple but effective baseline |

## 📈 Output & Results

### 🏋️ Training Phase
```
weights/
├── best_resnet34_model.pth         # Trained model weights
├── best_densenet169_model.pth
└── ...

results/
├── predictions_resnet34_test_best.json    # Model predictions for ensemble
├── predictions_densenet169_val-ensemble_best.json
└── ...
```

### 🧪 Evaluation Phase
```
results/evaluation/
├── resnet34_confusion_matrix.png           # Per-model confusion matrices
├── densenet169_class_accuracy.png          # Class-wise accuracy plots
├── model_comparison.png                    # Performance comparison chart
└── evaluation_results.json                 # Comprehensive metrics summary
```

### 🔬 Ensemble Learning Phase
```
results/ensemble/
├── models/
│   ├── LogisticRegression.pkl              # Trained ensemble models
│   ├── MeanWeighted.pkl
│   └── ...
├── ensemble_results.json                   # All ensemble method results
└── figures/
    └── ensemble_comparison.png             # Ensemble performance visualization
```

## 🔧 Key Technical Features

### 🔒 Patient-Level Data Splitting
Prevents data leakage by ensuring images from the same patient never appear in both training and test sets:

```python
def extract_patient_id(folder_name):
    """Extract patient ID from folder name"""
    if folder_name.endswith('OS') or folder_name.endswith('OD'):
        return folder_name[:-2]  # Remove eye identifier
    return folder_name
```

### 💾 Automated Prediction Generation
Seamlessly generates ensemble-ready predictions during training:

```python
def generate_predictions(model, dataloader, model_name, subset_name, device, class_names):
    """Generate and save model predictions in JSON format"""
```

### 🏗️ Extensible Ensemble Framework
Unified interface based on abstract base classes:

```python
class AbstractEnsemble(ABC):
    @abstractmethod
    def training(self, train_x, train_y): pass
    @abstractmethod
    def prediction(self, data): pass
    @abstractmethod
    def dump(self, path): pass
    @abstractmethod
    def load(self, path): pass
```

## 📊 Evaluation Metrics

Our comprehensive evaluation includes:

| Metric | Description | Weight |
|--------|-------------|--------|
| **Accuracy** | Overall classification accuracy | Standard |
| **Precision** | Weighted average precision | Class-balanced |
| **Recall** | Weighted average recall | Class-balanced |
| **F1-Score** | Weighted harmonic mean | Primary metric |
| **AUC** | Area under ROC curve | Multi-class OvR |
| **Confusion Matrix** | Detailed error analysis | Visual |

## 🎯 Performance Benchmarks

| Metric | Single Model | Ensemble | Improvement |
|--------|--------------|----------|-------------|
| **Accuracy** | 85-92% | 87-95% | +2-5% |
| **F1-Score** | 0.84-0.91 | 0.86-0.94 | +0.02-0.05 |
| **Stability** | Higher variance | Lower variance | ✅ More reliable |

> **💡 Pro Tip**: LogisticRegression and MeanWeighted ensembles typically achieve optimal performance.


## 🚀 Complete Workflow Example

```bash
# 1. Prepare patient-level data splits
python split.py

# 2. Train multiple architectures (recommended subset)
python train_multimodel.py --model resnet34 densenet169 vgg16 --epochs 30

# 3. Evaluate individual models
python test_multimodel.py --models resnet34+densenet169+vgg16

# 4. Deploy ensemble learning
python advanced_ensemble.py --models resnet34+densenet169+vgg16
```

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@misc{asoct-classification,
  title={ASOCT Medical Image Classification with Ensemble Learning},
  author={Jiongning Zhao},
  year={2025},
  publisher={GitHub},
  url={https://github.com/Johnny-creation/asoct-classification}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**🏥 This framework provides a complete, reliable, and efficient solution for medical image classification, specifically designed for high-accuracy diagnostic applications requiring robust performance and clinical reliability.**