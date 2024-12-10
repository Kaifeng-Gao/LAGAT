# 🕵️‍♂️ LAGAT 
> Label-Aware Graph Attention Network for Fraud Detection

LAGAT is a specialized Graph Neural Network designed for fraud detection in low-homophily settings. It enhances traditional GAT by incorporating neighborhood label information into the attention mechanism.

[LAGAT: Label-Aware Graph Attention Network for Fraud Detection](LAGAT.pdf)

## ✨ Key Features
- Label-aware attention mechanism for better information aggregation
- Optimized for low-homophily fraud detection scenarios

## 🏗️ Project Structure
```
LAGAT/
├── config/
│   └── config.py
├── data/
│   ├── mag/
│   ├── dataset_eda.ipynb
│   ├── fraud_dataset.py
│   └── load_fraud_dataset.py
├── model/
│   ├── gat.py
│   ├── gcn.py
│   ├── lagat.py
│   └── lagatconv.py
├── utils/
│   ├── earlystopping.py
│   ├── logging.py
│   ├── metrics.py
│   └── random_seed.py
├── visualization/
├── evaluate.py
├── LAGAT.pdf
├── README.md
├── result.ipynb
├── train.py
├── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
This repo has only been tested under
- Python 3.12
- CUDA 12.1

### Setting up Virtual Environment

Clone the repository
```bash
git clone https://github.com/Kaifeng-Gao/LAGAT.git
cd LAGAT
```

Create a virtual environment
```bash
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

Install required packages
```bash
# Install dependencies
pip install -r requirements.txt
```

## 🛠️ Usage

1. Configure training parameters in `config/config.py`
2. All hyperparameter combinations in `PARAM_GRID` will be trained sequentially

Example configuration:

```python
PARAM_GRID = {
    "dataset_config": [
        # dataset_name, target_node_type, in_channels
        ("yelp", "review", 32),
        ("amazon", "user", 25)
    ],
    "model_name": [
        "LAGAT", 
        "GAT"
    ],
    "train_size": [0.6],
    "val_size": [0.2],
    "observed_ratio": [0, 0.2, 0.4, 0.6, 0.8, 1],
    "random_seed": [42, 43, 44],
    "force_reload": [False],
    "epochs": [10000],
    "toleration": [300],
    "learning_rate": [0.005],
    "weight_decay": [5e-4],
    "hidden_channels": [32],
    "label_embedding_dim": [8],
    "num_layers": [2],
    "num_labels": [3],
    "out_channels": [2],
    "heads": [2],
    "dropout": [0.6]
}
```

Run training:

```bash
python train.py --use_wandb
# or
python train.py
```

## 📊 Results

LAGAT shows improved performance across multiple metrics while maintaining similar convergence rates to baseline models.

![result](/visualization/result.png)

### 🛍️ Amazon Dataset Results

| Model | Test AP (%) | Test AUC (%) | Test F1 (%) |
|-------|-------------|--------------|-------------|
| GAT   | 11.56 ± 2.36 | **55.23 ± 4.58** | 44.28 ± 16.15 |
| LAGAT | **14.59 ± 3.54** | 55.02 ± 3.51 | **55.32 ± 3.69** |

### 🍽️ Yelp Dataset Results

| Model | Test AP (%) | Test AUC (%) | Test F1 (%) |
|-------|-------------|--------------|-------------|
| GAT   | 28.59 ± 2.06 | 70.90 ± 3.28 | 46.12 ± 0.09 |
| LAGAT | **31.56 ± 1.38** | **73.45 ± 0.33** | **46.25 ± 0.21** |

