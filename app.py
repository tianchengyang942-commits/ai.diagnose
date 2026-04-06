import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
from safetensors.torch import load_file  # ⚠️ 使用了工业级安全读取


# ================= 1. 深度 UI 自定义 (CSS 魔法) =================
def set_page_style():
    st.set_page_config(page_title="Fetal Cleft AI", page_icon="🩺", layout="wide")  # 改为 wide 更有看片感

    st.markdown("""
        <style>
            /* 防止所有 Metric 的 label 溢出 */
            [data-testid="stMetricLabel"] > div {
                word-wrap: normal !important;
                white-space: normal !important;
                overflow: visible !important;
                display: -webkit-box !important;
                -webkit-line-clamp: 2; /* 限制两行 */
                -webkit-box-orient: vertical;
            }

            /* 防止 Metric 的 value 溢出 */
            [data-testid="stMetricValue"] > div {
                white-space: normal !important;
                overflow: visible !important;
            }
        </style>
    """, unsafe_allow_html=True)


set_page_style()


# ================= 2. 缓存加载两个神经网络 =================
@st.cache_resource
def load_ai_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. 检测器 (依然是 .pth，因为我们没改它)
    detector = models.resnet18(weights=None)
    detector.fc = nn.Linear(detector.fc.in_features, 4)
    detector.load_state_dict(torch.load('models/fetal_palate_detector.pth', map_location=device, weights_only=True))
    detector.to(device).eval()

    # 2. 分类器 (⚠️ 使用你刚刚跑出来的大图安全版 safetensors)
    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    classifier_state_dict = load_file('models/best_cleft_classifier.safetensors', device="cpu")
    classifier.load_state_dict(classifier_state_dict)
    classifier.to(device).eval()

    return detector, classifier, device


# ================= 3. 仿生诊断引擎实现 =================
def run_diagnosis_pipeline(original_image, detector, classifier, device):
    orig_w, orig_h = original_image.size

    # 阶段一：目标检测定位靶区
    det_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    det_input = det_transform(original_image.resize((256, 256))).unsqueeze(0).to(device)
    with torch.no_grad():
        coords = detector(det_input)[0].tolist()

    scale_w, scale_h = orig_w / 256.0, orig_h / 256.0
    xmin, ymin = max(0, coords[0] * scale_w), max(0, coords[1] * scale_h)
    xmax, ymax = min(orig_w, coords[2] * scale_w), min(orig_h, coords[3] * scale_h)

    # 阶段二：靶区特征鉴别 (退回“两步走”架构：先切图，再诊断)
    # 🔪 关键修复：根据定位坐标，把上腭靶区切成小方块！
    palate_crop = original_image.crop((xmin, ymin, xmax, ymax))

    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 恢复老模型熟悉的 224x224 分辨率
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 将切下来的小方块送给分类器进行判断
    cls_input = cls_transform(palate_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(cls_input)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # {'Abnormal_Or_NotPalate': 0, 'NormalPalate': 1}
    prob_normal, prob_cleft = probabilities[1].item(), probabilities[0].item()

    # 阶段三：报告生成
    # 复制一张图用来画框，防止污染原图
    result_image = original_image.copy()
    draw = ImageDraw.Draw(result_image)
    threshold = 0.60

    if prob_normal > threshold:
        status, color = "NORMAL PALATE", "#00FF00"
    else:
        status, color = "CLP SUSPECTED", "#FF0000"

    # 画出定位框供医生参考
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)

    return result_image, status, prob_normal, prob_cleft


# ================= 4. 主界面：混合布局设计 =================
detector, classifier, device = load_ai_models()

st.title("🏥 胎儿唇腭裂 AI 全景辅助诊断系统")
st.markdown("---")

# ----------------- A. 侧边栏上传区 -----------------
uploaded_file = st.sidebar.file_uploader("📁 请选择本地超声图像 (jpg/png)", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.markdown("💡 **操作提示**：\n您可以选择上传本地文件，或者直接点击右侧画廊中的示例图片体验。")

# ----------------- B. 快速体验画廊区 -----------------
# 只有当用户没有上传文件时，才显示示例画廊
image_to_analyze = None

if uploaded_file is None:
    st.subheader("💡 快速体验示例")

    # 确保你的文件夹和图片是存在的，否则按钮会报错
    EXAMPLE_IMAGES = {
        "🌟 示例 1: 正常胎儿上腭": "examples/normal_sample.png",
        "⚠️ 示例 2: 疑似唇裂病变": "examples/cleft_sample.png"
    }
    os.makedirs("examples", exist_ok=True)

    cols = st.columns(max(len(EXAMPLE_IMAGES), 1))

    for i, (name, path) in enumerate(EXAMPLE_IMAGES.items()):
        with cols[i]:
            if os.path.exists(path):
                st.image(path, caption=name, use_container_width=True)
                # 使用按钮直接触发诊断
                if st.button(f"一键智能诊断：{name.split(':')[0]}", key=f"btn_{i}"):
                    image_to_analyze = Image.open(path).convert('RGB')
            else:
                st.warning(f"缺少示例图: {path}")
    st.markdown("---")

else:
    # 用户上传了文件
    image_to_analyze = Image.open(uploaded_file).convert('RGB')

# ----------------- C. 诊断报告核心区 -----------------
# 只要 image_to_analyze 有图片（无论来自上传还是示例），就进入看片模式
if image_to_analyze is not None:
    st.markdown("### 📊 影像科 AI 辅助筛查报告")
    st.markdown("---")

    with st.spinner("🚀 双核端到端引擎正在深挖纹理特征..."):
        # 1. 后台秒级运算
        processed_image, diagnosis, prob_n, prob_c = run_diagnosis_pipeline(
            image_to_analyze, detector, classifier, device
        )

    # 2. 极简分列对齐
    col1, col2 = st.columns([6, 4])

    with col1:
        # 左侧放图
        st.image(processed_image, use_container_width=True)

    with col2:
        # 右侧：彻底抛弃容易错位的 st.metric，改用定制 HTML 医疗数据卡片
        if diagnosis == "NORMAL PALATE":
            st.success("🏆 最终结论: **NORMAL (正常)**", icon="✅")
            border_color = "#28a745" # 正常为绿色边框
        else:
            st.error("⚠️ 最终结论: **CLP SUSPECTED (疑似唇腭裂)**", icon="🚨")
            border_color = "#dc3545" # 异常为红色边框

        # 直接渲染一个美观、紧凑且高度绝对受控的数据看板
        st.markdown(f"""
        <div style="
            background-color: #f8f9fa; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 6px solid {border_color};
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-top: 10px;
        ">
            <h4 style="margin-top: 0; color: #333;">🔬 核心诊断指标</h4>
            <div style="font-size: 16px; color: #555; line-height: 1.8;">
                <div>✅ 正常解剖结构匹配度：<strong style="color:#333;">{prob_n * 100:.1f}%</strong></div>
                <div>❌ 病理声影特征匹配度：<strong style="color:#333;">{prob_c * 100:.1f}%</strong></div>
            </div>
            <hr style="border: 0; border-top: 1px solid #ddd; margin: 15px 0;">
            <div style="font-size: 18px; color: #111;">
                🧠 综合诊断置信度：<strong>{max(prob_n, prob_c) * 100:.1f}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)