import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont

# ================= 1. 配置路径与模型配置 =================
DETECTOR_WEIGHTS = 'fetal_palate_detector.pth'
# 注意加载你刚刚新训练出来的那个“大脑”
CLASSIFIER_WEIGHTS = 'best_cleft_classifier.pth'

# 用于测试的原始图片
TEST_IMAGE_PATH = '/Users/tianchengyang/Desktop/Synthetic_Robust_CLP_Dataset/Class_0_Abnormal_Or_NotPalate/cleft_15.png'
OUTPUT_REPORT_PATH = 'examples/final_clinical_report.jpg'

# 如果你的 Mac 安装了中文字体，可以用，否则默认英文
FONT_PATH = '/System/Library/Fonts/Supplemental/Arial.ttf'  # 默认英文字体


def load_models(device):
    print("⏳ 正在加载双核医用 AI 引擎...")

    # 1. 检测器
    detector = models.resnet18(weights=None)
    detector.fc = nn.Linear(detector.fc.in_features, 4)
    detector.load_state_dict(torch.load(DETECTOR_WEIGHTS, map_location=device, weights_only=True))
    detector.to(device);
    detector.eval()

    # 2. 分类器
    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=device, weights_only=True))
    classifier.to(device);
    classifier.eval()

    return detector, classifier


if __name__ == "__main__":
    print("\n🏥 胎儿唇腭裂全自动 AI 诊断系统 (医学增强版) 🏥")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    detector, classifier = load_models(device)

    if not os.path.exists(TEST_IMAGE_PATH):
        TEST_IMAGE_PATH = TEST_IMAGE_PATH.replace('.png', '.jpg')
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ 找不到测试图片: {TEST_IMAGE_PATH}");
        exit()

    original_image = Image.open(TEST_IMAGE_PATH).convert('RGB')
    orig_w, orig_h = original_image.size

    # -------- 阶段一：靶区精准定位 (Detector) --------
    print("🔍 [1/3] 检测器正在寻找上腭解剖靶区...")
    det_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    det_input = det_transform(original_image.resize((256, 256))).unsqueeze(0).to(device)
    with torch.no_grad():
        coords = detector(det_input)[0].tolist()

    # 坐标放大还原 (不再外扩，只裁靶区)
    scale_w, scale_h = orig_w / 256.0, orig_h / 256.0
    xmin, ymin = max(0, coords[0] * scale_w), max(0, coords[1] * scale_h)
    xmax, ymax = min(orig_w, coords[2] * scale_w), min(orig_h, coords[3] * scale_h)

    # 安全性检查：如果检测出的框面积太小，可能定位失败
    if (xmax - xmin) * (ymax - ymin) < 20 * 20:
        print("❌ AI 无法在全图中定位明确的上腭结构，诊断终止。")
        original_image.save('failed_diagnosis.jpg')
        exit()

    # -------- 阶段二：靶区特征分析 (Classifier) --------
    print("🔬 [2/3] 分类器正在对靶区纹理进行病理分析...")
    # 切割精准靶区
    palate_crop = original_image.crop((xmin, ymin, xmax, ymax))

    cls_transform = transforms.Compose([
        # 裁剪的靶区比例可能不同，统一强制缩放
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    cls_input = cls_transform(palate_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(cls_input)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # 字典 {'Abnormal_Or_NotPalate': 0, 'NormalPalate': 1}
    prob_cleft = probabilities[0].item() * 100
    prob_normal = probabilities[1].item() * 100

    # -------- 阶段三：出具诊断报告 --------
    print("📄 [3/3] 分析完毕，正在生成诊断报告...")
    draw = ImageDraw.Draw(original_image)

    # 决策逻辑
    threshold = 60.0  # 提高门槛，高于 60% 正常才算正常

    if prob_normal > threshold:
        color = "#00FF00"  # 健康绿
        status = "Normal Palate"
        conf = prob_normal
    else:
        # 如果正常概率不高，判定为 Cleft 或 非腭部结构
        color = "red"  # 警告红
        status = "CLP Suspected / Not Palate"
        conf = max(prob_cleft, 100 - prob_normal)

    # 在原图画精准靶区框
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)

    # 把诊断写在框上方
    try:
        font = ImageFont.truetype(FONT_PATH, 24)
    except:
        font = ImageFont.load_default()

    text = f"{status} ({conf:.1f}%)"
    draw.text((xmin, max(0, ymin - 30)), text, fill=color, font=font)

    print("\n" + "=" * 40)
    print(f"🩺 最终诊断: {status} (置信度 {conf:.1f}%)")
    print(f"📊 详情: 正常腭部({prob_normal:.1f}%), 疑似唇裂/干扰器官({prob_cleft:.1f}%)")
    print("=" * 40)

    original_image.save(OUTPUT_REPORT_PATH)
    print(f"🎉 带有医学标注的诊断报告已保存: {OUTPUT_REPORT_PATH}")