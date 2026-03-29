import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
import os

# ================= 1. 配置路径与参数 =================
# 指向我们刚刚在桌面上生成的合成数据集
DATASET_DIR = '/Users/tianchengyang/Desktop/Synthetic_Robust_CLP_Dataset'

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001


# ================= 2. 数据准备与增强 =================
def prepare_data():
    # 数据预处理：ResNet 默认最喜欢 224x224 的输入
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 自动从文件夹名称提取分类 (Cleft 和 Normal)
    full_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=data_transforms)

    # 打印类别对应的数字标签 (比如 {'Cleft': 0, 'Normal': 1})
    print(f"🏷️ AI 自动识别的疾病标签字典: {full_dataset.class_to_idx}")

    # 划分数据集：80% 训练，20% 考试验证
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  # 考试不需要打乱

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


# ================= 3. 构建二分类模型 =================
def create_classifier():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    # 我们只有 2 个类别：Cleft(唇裂) 和 Normal(正常)
    model.fc = nn.Linear(num_ftrs, 2)
    return model


# ================= 4. 主训练流程 =================
if __name__ == "__main__":
    print("🚀 正在启动 AI 唇腭裂辅助诊断系统 (分类训练)...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ 计算设备: {device}")

    # 加载数据
    train_loader, val_loader, train_size, val_size = prepare_data()
    print(f"📚 学习资料准备完毕！训练集: {train_size} 张，考试验证集: {val_size} 张。")

    # 初始化模型、损失函数和优化器
    model = create_classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0  # 记录最高准确率

    # 开始循环训练
    for epoch in range(EPOCHS):
        print(f"\n--- 第 {epoch + 1}/{EPOCHS} 轮学习开始 ---")

        # 1. 训练阶段 (AI 看书学习)
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / train_size

        # 2. 验证阶段 (AI 闭卷考试)
        model.eval()
        corrects = 0

        with torch.no_grad():  # 考试时不能翻书（不计算梯度）
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # 选出概率最大的那个类别作为最终答案
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)

        epoch_acc = corrects.float() / val_size

        print(f"📉 训练误差 (Loss): {epoch_loss:.4f}")
        print(f"🎯 诊断准确率 (Accuracy): {epoch_acc * 100:.2f}%")

        # 如果这次考试成绩是历史最高，就把这个“最强大脑”保存下来
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), '../models/best_cleft_classifier.pth')
            print("🌟 创造了新的准确率纪录！已保存模型！")

    print(f"\n🎉 训练全部结束！最高诊断准确率达到了: {best_acc * 100:.2f}%")
    print("💾 最终的医疗 AI 引擎已保存至: best_cleft_classifier.pth")