import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
# 导入我们刚刚改好的、能输出 4 个坐标的数据集类
from scripts.dataset import FetalFacialDataset

# ================= 1. 路径与参数配置 =================
ANNOTATION_PATH = '/Users/tianchengyang/Desktop/Dataset for Fetus Framework/data/ObjectDetection.xlsx'
IMG_DIR = '/Users/tianchengyang/Desktop/Dataset for Fetus Framework/data/images'

BATCH_SIZE = 8  # 找坐标的任务稍微难一点，我们把 Batch 调小一点让它学得更仔细
LEARNING_RATE = 0.001
EPOCHS = 10  # 先跑 10 轮看看误差下降趋势


# ================= 2. 构建目标检测模型 =================
def create_detection_model():
    # 依然站在巨人的肩膀上，加载 ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 提取它的全连接层输入维度
    num_ftrs = model.fc.in_features

    # ⭐️ 核心改变 1：输出 4 个神经元，分别代表预测的 4 个坐标值
    model.fc = nn.Linear(num_ftrs, 4)
    return model


# ================= 3. 主训练流程 =================
if __name__ == "__main__":
    print("🚀 正在启动 AI 目标检测(寻找腭部坐标)训练程序...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ 当前使用的计算设备: {device}")

    # 数据预处理 (注意：这里不要加 RandomCrop 等会改变图片几何形状的操作，否则坐标就全错位了)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    dataset = FetalFacialDataset(annotation_path=ANNOTATION_PATH, img_dir=IMG_DIR, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型
    model = create_detection_model().to(device)

    # ⭐️ 核心改变 2：使用 Smooth L1 Loss 计算坐标回归误差
    # 相比于 MSE，它对异常值（离得太远的错误预测）更具包容性，是目标检测任务的标配
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"📚 开始训练寻找坐标，总图片量: {len(dataset)}，共需跑 {EPOCHS} 轮...")

    # 开始循环训练
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # 注意这里接收的是 bboxes (真实坐标矩阵)
        for i, (inputs, bboxes) in enumerate(dataloader):
            inputs = inputs.to(device)
            # 确保真实坐标是 float 类型，才能和模型输出计算误差
            bboxes = bboxes.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            # 预测出 4 个坐标
            predicted_bboxes = model(inputs)

            # 计算预测坐标和真实坐标相差多少个像素单位
            loss = criterion(predicted_bboxes, bboxes)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 每 10 个 Batch 打印一次进度
            if (i + 1) % 10 == 0:
                print(f"   [Epoch {epoch + 1}/{EPOCHS}, Batch {i + 1}] 坐标误差 (Loss): {running_loss / 10:.4f}")
                running_loss = 0.0

    print("🎉 恭喜！模型试跑完成！你的 AI 现在不仅会认，还会找了！")

    # 保存会找坐标的新大脑
    torch.save(model.state_dict(), '../models/fetal_palate_detector.pth')
    print("💾 目标检测模型已成功保存到 fetal_palate_detector.pth！")