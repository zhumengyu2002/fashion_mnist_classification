# utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def plot_sample_images(images, labels, class_names, num_samples=10):
    """绘制样本图像"""
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(class_names[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, train_accs=None, val_accs=None):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=(12, 4))

    if train_accs:
        axes[0].plot(train_losses, label='Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(train_accs, label='Training Accuracy')
        if val_accs:
            axes[1].plot(val_accs, label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes.plot(train_losses, label='Training Loss')
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        axes.set_title('Training Loss')
        axes.legend()
        axes.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def print_classification_report(y_true, y_pred, class_names):
    """打印分类报告"""
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)