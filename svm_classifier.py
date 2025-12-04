# svm_classifier.py
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import time
from utils import plot_confusion_matrix, print_classification_report
import matplotlib.pyplot as plt
...


class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        初始化SVM分类器

        参数:
            kernel: 核函数 ('linear', 'rbf', 'poly', 'sigmoid')
            C: 正则化参数
            gamma: RBF核的参数
        """
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def preprocess_data(self, X_train, X_test):
        """数据预处理：归一化并展平"""
        # 归一化到 [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0

        # 展平图像
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

        return X_train_flat, X_test_flat

    def train(self, X_train, y_train):
        """训练SVM模型"""
        print("开始训练SVM模型...")
        start_time = time.time()

        self.model.fit(X_train, y_train)

        train_time = time.time() - start_time
        print(f"SVM训练完成，耗时: {train_time:.2f}秒")

        # 训练集准确率
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"训练集准确率: {train_acc:.4f}")

        return train_acc

    def evaluate(self, X_test, y_test):
        """评估模型性能"""
        print("\n开始测试SVM模型...")
        start_time = time.time()

        y_pred = self.model.predict(X_test)
        test_time = time.time() - start_time

        test_acc = accuracy_score(y_test, y_pred)
        print(f"测试集准确率: {test_acc:.4f}")
        print(f"测试耗时: {test_time:.2f}秒")

        # 打印分类报告
        print("\n分类报告:")
        print_classification_report(y_test, y_pred, self.class_names)

        # 绘制混淆矩阵
        plot_confusion_matrix(y_test, y_pred, self.class_names)

        return test_acc, y_pred

    def compare_kernels(self, X_train, y_train, X_test, y_test, kernels=['linear', 'rbf']):
        """比较不同核函数的性能"""
        results = {}

        for kernel in kernels:
            print(f"\n正在训练 {kernel.upper()} 核SVM...")
            model = svm.SVC(kernel=kernel, C=1.0, gamma='scale', random_state=42)

            # 训练
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # 预测
            y_pred = model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)

            results[kernel] = {
                'accuracy': test_acc,
                'train_time': train_time
            }

            print(f"{kernel.upper()} 核 - 测试准确率: {test_acc:.4f}, 训练时间: {train_time:.2f}秒")

        # 绘制结果对比
        kernels_list = list(results.keys())
        accuracies = [results[k]['accuracy'] for k in kernels_list]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(kernels_list, accuracies, color=['skyblue', 'lightcoral'])
        plt.xlabel('核函数')
        plt.ylabel('准确率')
        plt.title('不同核函数的SVM性能对比')
        plt.ylim([0.7, 0.9])

        # 在柱状图上显示准确率数值
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{acc:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        return results