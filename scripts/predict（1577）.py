import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw

# ================= 1. 配置路径 =================
MODEL_WEIGHTS_PATH = '../models/fetal_palate_detector.pth'

# 测试刚刚那张 1577 的图片
TEST_IMAGE_PATH = '/Users/tianchengyang/Desktop/Dataset for Fetus Framework/Dataset for Fetus Framework/Internal Test Set/Standard/1577.png'
OUTPUT_IMAGE_PATH = '../examples/prediction_result_1577.jpg'


# ================= 2. 重建目标检测模型 =================
def load_detector_model(weights_path, device):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    print("🏥 AI 影像科医生已就位，正在还原大图坐标...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_detector_model(MODEL_WEIGHTS_PATH, device)

    if not os.path.exists(TEST_IMAGE_PATH):
        TEST_IMAGE_PATH = TEST_IMAGE_PATH.replace('.png', '.jpg')

    # 1. 打开原始大图并记录真实宽高
    original_image = Image.open(TEST_IMAGE_PATH).convert('RGB')
    orig_w, orig_h = original_image.size
    print(f"📸 原始超声图分辨率为: 宽 {orig_w} x 高 {orig_h}")

    # 2. 缩小给 AI 看
    resized_image = original_image.resize((256, 256))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(resized_image).unsqueeze(0).to(device)

    # 3. AI 在 256x256 上预测出的坐标
    with torch.no_grad():
        predicted_coords = model(input_tensor)[0]
    xmin, ymin, xmax, ymax = predicted_coords.tolist()

    # ⭐️ 4. 核心魔法：将 AI 预测的坐标放大回原始比例 ⭐️
    scale_w = orig_w / 256.0
    scale_h = orig_h / 256.0

    real_xmin = xmin * scale_w
    real_ymin = ymin * scale_h
    real_xmax = xmax * scale_w
    real_ymax = ymax * scale_h

    print(
        f"✅ 还原后的原图预测坐标为: xmin={real_xmin:.1f}, ymin={real_ymin:.1f}, xmax={real_xmax:.1f}, ymax={real_ymax:.1f}")

    # 5. 直接在原始大图上画框！
    draw = ImageDraw.Draw(original_image)

    # 画 AI 预测的红框
    draw.rectangle([real_xmin, real_ymin, real_xmax, real_ymax], outline="red", width=4)

    # （可选）把医生在 Excel 里的标准答案用绿框画出来对比
    # 根据我查到的数据 1577 的真实坐标是 317, 169, 407, 219
    draw.rectangle([317, 169, 407, 219], outline="green", width=4)

    original_image.save(OUTPUT_IMAGE_PATH)
    print(f"🎉 诊断完毕！红框为 AI 预测，绿框为医生标注。请打开 {OUTPUT_IMAGE_PATH} 对比！")