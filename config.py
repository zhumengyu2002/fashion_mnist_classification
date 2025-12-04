# config.py
import torch

# 数据集配置
DATASET_PATH = "./data"
NUM_CLASSES = 10
IMG_SIZE = 28

# SVM配置
SVM_CONFIG = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale'
}

# CNN配置
CNN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'epochs': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}