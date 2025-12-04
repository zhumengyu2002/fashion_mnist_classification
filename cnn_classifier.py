# cnn_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from tqdm import tqdm

from cnn_model import EnhancedCNN, LeNet5
from utils import plot_training_history, plot_confusion_matrix, print_classification_report


class CNNClassifier:
    def __init__(self, model_type='lenet5', num_classes=10, device='cpu'):
        """
        初始化CNN分类器

        参数:
            model_type: 模型类型 ('lenet5' 或 'enhanced')
            num_classes: 类别数量
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # 选择模型
        if model_type == 'lenet5':
            self.model = LeNet5(num_classes=num_classes)
        elif model_type == 'enhanced':
            self.model = EnhancedCNN(num_classes=num_classes)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        self.model.to(self.device)
        print(f"使用模型: {model_type}")
        print(f"使用设备: {device}")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def preprocess_data(self, X_train, y_train, X_test, y_test, batch_size=64):
        """数据预处理并创建数据加载器"""
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(1)  # 增加通道维度
        y_train_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1)
        y_test_tensor = torch.LongTensor(y_test)

        # 归一化到 [0, 1]
        X_train_tensor = X_train_tensor / 255.0
        X_test_tensor = X_test_tensor / 255.0

        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def train(self, train_loader, val_loader=None, epochs=10, lr=0.001):
        """训练CNN模型"""
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        train_losses = []
        train_accuracies = []
        val_accuracies = []

        print("开始训练CNN模型...")

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # 使用tqdm显示进度条
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

            # 计算平均损失和准确率
            avg_loss = running_loss / len(train_loader)
            train_acc = 100. * correct / total
            train_losses.append(avg_loss)
            train_accuracies.append(train_acc)

            # 验证集评估
            val_acc = 0
            if val_loader:
                val_acc = self.evaluate(val_loader)
                val_accuracies.append(val_acc)
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, '
                      f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            else:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, '
                      f'Train Acc: {train_acc:.2f}%')

            # 更新学习率
            scheduler.step()

        # 绘制训练历史
        plot_training_history(train_losses, train_accuracies,
                              val_accuracies if val_loader else None)

        return train_losses, train_accuracies, val_accuracies if val_loader else None

    def evaluate(self, test_loader):
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)

                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = 100. * correct / total
        print(f'\n测试集准确率: {accuracy:.2f}%')
        print(f'正确预测数: {correct}/{total}')

        # 打印分类报告
        print("\n分类报告:")
        print_classification_report(all_targets, all_preds, self.class_names)

        # 绘制混淆矩阵
        plot_confusion_matrix(all_targets, all_preds, self.class_names)

        return accuracy, all_preds

    def predict(self, images):
        """预测单个或批量图像"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(images, np.ndarray):
                images = torch.FloatTensor(images).unsqueeze(1).to(self.device)
            images = images.to(self.device)
            outputs = self.model(images)
            _, predicted = outputs.max(1)
        return predicted.cpu().numpy()

    def save_model(self, path='cnn_model.pth'):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': 'lenet5' if isinstance(self.model, LeNet5) else 'enhanced',
            'num_classes': self.num_classes
        }, path)
        print(f"模型已保存到: {path}")

    def load_model(self, path='cnn_model.pth'):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {path} 加载")