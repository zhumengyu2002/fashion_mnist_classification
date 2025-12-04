# cnn_model.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet-5风格的CNN模型"""

    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # 池化层
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 第一次池化后: (28-5+4)/2+1 = 14, 第二次池化后: (14-5+0)/2+1 = 5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.conv1(x)))  # [N, 1, 28, 28] -> [N, 6, 14, 14]

        # 第二个卷积块
        x = self.pool(F.relu(self.conv2(x)))  # [N, 6, 14, 14] -> [N, 16, 5, 5]

        # 展平
        x = x.view(-1, 16 * 5 * 5)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class EnhancedCNN(nn.Module):
    """增强的CNN模型，包含更多卷积层"""

    def __init__(self, num_classes=10):
        super(EnhancedCNN, self).__init__()

        # 卷积块1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # 卷积块2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 卷积块3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # 经过3次池化: 28 -> 14 -> 7 -> 3
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积块1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # [N, 1, 28, 28] -> [N, 32, 14, 14]

        # 卷积块2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # [N, 32, 14, 14] -> [N, 64, 7, 7]

        # 卷积块3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # [N, 64, 7, 7] -> [N, 128, 3, 3]

        # 展平
        x = x.view(-1, 128 * 3 * 3)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x