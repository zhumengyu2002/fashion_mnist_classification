# main.py
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from svm_classifier import SVMClassifier
from cnn_classifier import CNNClassifier
from utils import plot_sample_images
import config
import matplotlib.pyplot as plt
import matplotlib

# 尝试多种解决方案
try:
    # 方案1：尝试使用系统中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    print("中文字体设置尝试完成")
except:
    # 方案2：如果失败，修改可视化函数使用英文
    print("中文字体设置失败，将使用英文显示")

def load_fashion_mnist():
    """加载Fashion-MNIST数据集"""
    print("正在加载Fashion-MNIST数据集...")

    # 使用torchvision加载数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下载训练集
    trainset = torchvision.datasets.FashionMNIST(
        root=config.DATASET_PATH,
        train=True,
        download=True,
        transform=transform
    )

    # 下载测试集
    testset = torchvision.datasets.FashionMNIST(
        root=config.DATASET_PATH,
        train=False,
        download=True,
        transform=transform
    )

    # 转换为numpy数组
    X_train = trainset.data.numpy()
    y_train = trainset.targets.numpy()
    X_test = testset.data.numpy()
    y_test = testset.targets.numpy()

    print(f"训练集形状: {X_train.shape}, 标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test.shape}, 标签形状: {y_test.shape}")

    # 类别名称
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return X_train, y_train, X_test, y_test, class_names


def run_svm_classification(X_train, y_train, X_test, y_test):
    """运行SVM分类"""
    print("\n" + "=" * 50)
    print("SVM图像分类")
    print("=" * 50)

    # 创建SVM分类器
    svm_classifier = SVMClassifier(kernel='rbf', C=1.0)

    # 数据预处理
    X_train_flat, X_test_flat = svm_classifier.preprocess_data(X_train, X_test)

    # 训练SVM
    train_acc = svm_classifier.train(X_train_flat, y_train)

    # 测试SVM
    test_acc, y_pred = svm_classifier.evaluate(X_test_flat, y_test)

    # 比较不同核函数
    print("\n" + "=" * 50)
    print("不同核函数SVM性能比较")
    print("=" * 50)
    kernel_results = svm_classifier.compare_kernels(
        X_train_flat[:10000], y_train[:10000],  # 使用部分数据加速
        X_test_flat[:2000], y_test[:2000],
        kernels=['linear', 'rbf']
    )

    return test_acc, kernel_results


def run_cnn_classification(X_train, y_train, X_test, y_test, device='cpu'):
    """运行CNN分类"""
    print("\n" + "=" * 50)
    print("CNN图像分类")
    print("=" * 50)

    # 创建CNN分类器
    cnn_classifier = CNNClassifier(model_type='lenet5', device=device)

    # 数据预处理并创建数据加载器
    train_loader, test_loader = cnn_classifier.preprocess_data(
        X_train, y_train, X_test, y_test,
        batch_size=config.CNN_CONFIG['batch_size']
    )

    # 训练CNN
    train_losses, train_accs, _ = cnn_classifier.train(
        train_loader,
        epochs=config.CNN_CONFIG['epochs'],
        lr=config.CNN_CONFIG['learning_rate']
    )

    # 测试CNN
    test_acc, y_pred = cnn_classifier.evaluate(test_loader)

    # 保存模型
    cnn_classifier.save_model('fashion_mnist_cnn.pth')

    return test_acc


def visualize_samples(X_train, y_train, class_names):
    """可视化样本"""
    print("\n" + "=" * 50)
    print("数据集样本可视化")
    print("=" * 50)

    # 显示前10个样本
    plot_sample_images(X_train[:10], y_train[:10], class_names)

    # 显示每个类别的样本分布
    unique, counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts, color='skyblue')
    plt.xticks(unique, [class_names[i] for i in unique], rotation=45, ha='right')
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    plt.title('训练集类别分布')
    plt.tight_layout()
    plt.show()


def compare_results(svm_acc, cnn_acc, kernel_results):
    """对比SVM和CNN的结果"""
    print("\n" + "=" * 50)
    print("SVM与CNN性能对比")
    print("=" * 50)

    # 创建对比表格
    results = {
        'Method': ['SVM (RBF)', 'SVM (Linear)', 'CNN (LeNet-5)'],
        'Accuracy': [
            kernel_results.get('rbf', {}).get('accuracy', 0) if kernel_results else svm_acc,
            kernel_results.get('linear', {}).get('accuracy', 0) if kernel_results else 0,
            cnn_acc / 100  # CNN准确率是百分比，转换为小数
        ],
        'Training Time': [
            kernel_results.get('rbf', {}).get('train_time', 0) if kernel_results else 0,
            kernel_results.get('linear', {}).get('train_time', 0) if kernel_results else 0,
            '~5-10 minutes'  # CNN训练时间估计
        ]
    }

    # 打印表格
    print(f"{'Method':<20} {'Accuracy':<15} {'Training Time':<20}")
    print("-" * 55)
    for i in range(len(results['Method'])):
        acc = results['Accuracy'][i]
        time_val = results['Training Time'][i]
        print(f"{results['Method'][i]:<20} {acc:.4f} ({acc * 100:.2f}%){'' if i == 2 else '':<6} {str(time_val):<20}")

    # 绘制准确率对比图
    methods = results['Method']
    accuracies = results['Accuracy']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color=['lightcoral', 'lightgreen', 'skyblue'])
    plt.xlabel('方法')
    plt.ylabel('准确率')
    plt.title('不同分类方法在Fashion-MNIST上的性能对比')
    plt.ylim([0.7, 0.95])

    # 在柱状图上显示准确率数值
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{acc:.4f}\n({acc * 100:.2f}%)', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("Fashion-MNIST图像分类作业")
    print("=" * 50)

    # 1. 加载数据集
    X_train, y_train, X_test, y_test, class_names = load_fashion_mnist()

    # 2. 可视化样本
    visualize_samples(X_train, y_train, class_names)

    # 3. 运行SVM分类
    svm_acc, kernel_results = run_svm_classification(X_train, y_train, X_test, y_test)

    # 4. 运行CNN分类
    cnn_acc = run_cnn_classification(
        X_train, y_train, X_test, y_test,
        device=config.CNN_CONFIG['device']
    )

    # 5. 结果对比
    compare_results(svm_acc, cnn_acc, kernel_results)

    print("\n" + "=" * 50)
    print("作业完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()