# Fashion-MNIST图像分类：SVM与CNN的对比实验

## 1. 项目简介
本项目对比了传统机器学习方法与深度学习方法在图像分类任务上的性能。在 Fashion-MNIST 数据集上，分别实现了：
- 支持向量机 (SVM)：使用原始像素特征，对比了线性与RBF核函数。
- 卷积神经网络 (CNN)：实现了一个LeNet-5风格的网络结构。
通过对比两者的准确率、训练时间及混淆矩阵，分析其优缺点与适用场景。

## 2. 环境依赖
运行本项目需要以下Python库（主要版本）：
Python >= 3.8
torch >= 1.9.0 
torchvision >= 0.10.0
scikit-learn >= 0.24.2
matplotlib >= 3.3.4
numpy >= 1.19.5
tqdm >= 4.62.0

## 3.项目结构
```
fashion-mnist-classification/
├── README.md # 本文件
├── requirements.txt # 依赖包列表
├── main.py # 主程序入口，协调整个实验流程
├── config.py # 配置文件（参数设置）
├── svm_classifier.py # SVM分类器的实现与评估
├── cnn_classifier.py # CNN分类器的训练与评估
├── utils.py # 工具函数（可视化、报告生成等）
├── models/
│ └── cnn_model.py # LeNet-5等CNN模型定义
└── results/ # 存放运行时生成的图表（需自行创建）
├── svm_cm.png
├── cnn_training.png
└── model_comparison.png
```

## 4.运行步骤
（1）克隆项目并安装依赖：
git clone <你的项目仓库地址>
cd fashion-mnist-classification
pip install -r requirements.txt

（2）准备结果目录：创建 results/ 文件夹用于保存输出图片。

（3）运行主程序：python main.py
程序将自动：
- 下载Fashion-MNIST数据集
- 训练并评估SVM（Linear和RBF核）
- 训练并评估CNN（LeNet-5）
- 在控制台输出指标并生成可视化图表

## 5.结果展示
### 5.1 SVM与CNN性能对比
| 模型 | 测试准确率 | 训练时间 | 备注 |
| :--- | :--- | :--- | :--- |
| **SVM (RBF 核)** | 0.8600 (86.00%) | 约 6.17 秒 | 非线性核函数 |
| **SVM (Linear 核)** | 0.8305 (83.05%) | 约 5.53 秒 | 线性核函数 |
| **CNN (LeNet-5)** | **0.8877 (88.77%)** | 约 5-10 分钟 (GPU) | 卷积神经网络 |

> **不同方法性能对比图**：CNN取得了最高的分类准确率。
results/model_comparison.png

## 5.2 训练过程可视化
CNN训练过程：损失函数持续下降，准确率稳步上升，训练过程健康、收敛良好。
cnn_training.png

## 5.3 混淆矩阵分析
CNN混淆矩阵：模型在“T-shirt/top”、“Shirt”、“Pullover”、“Coat”、“Dress”等上衣类别间存在混淆， 这与它们在灰度图像中的视觉相似性一致。而“Trouser”、“Bag”、“Sandal”等特征独特的类别几乎被完美分类。
results/plot_2025-12-04 21-15-50_5.png


## 6.实验分析与总结
## 6.1关键发现
（1）特征表示决定性能上限：
- SVM：使用原始像素作为特征，性能受限于手工特征（核函数）的设计。RBF核通过非线性映射，比Linear核捕捉了更复杂的关系，性能显著提升。
- CNN：通过卷积层自动学习层次化的空间特征，直接从原始像素中提取了比单一像素向量更有效的表示，因此取得了最优性能。

（2）模型“困惑”具有可解释性：
混淆矩阵显示，所有模型都在相似的类别对上出错（如上衣类）。

（3）计算效率与性能的权衡：
- SVM：训练和预测速度极快，适合需要快速原型验证或对延迟敏感的场景。
- CNN：训练需要更多时间和计算资源（尤其是GPU），但能提供更高的准确率，适合对精度要求高且具备相应资源的场景。

## 6.2结论
本实验验证了从传统机器学习到深度学习的演进逻辑：
- 对于Fashion-MNIST这类相对复杂的图像分类任务，能够自动学习特征的CNN模型，在准确率上超越了依赖手工设计核函数的SVM模型。
- SVM作为强大的传统方法，在小数据集上仍能提供快速且具有一定竞争力的基线结果。
- 实验结果完全符合理论预期，训练过程稳定，模型行为可解释，整体实验成功。
