import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO


# ================= 1. 深度 UI 自定义 (CSS 魔法) =================
# 为了实现极简布局并防止文字溢出，我们直接注入一点 CSS
def set_page_style():
    st.set_page_config(page_title="Fetal Cleft AI", page_icon="🩺", layout="centered")

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

            /* 卡片布局的通用样式 */
            .medical-card {
                padding: 1.5rem;
                border-radius: 0.8rem;
                background-color: #f0f2f6; /* 干净的浅灰色背景 */
                border-left: 5px solid #0056b3; /* 医用蓝色边框 */
                margin-bottom: 1rem;
            }

            /* 调整 Metric 卡片样式 */
            [data-testid="stMetric"] {
                padding: 1rem;
                background-color: #ffffff;
                border-radius: 0.5rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
        </style>
    """, unsafe_allow_html=True)


set_page_style()


# ================= 2. 缓存加载两个神经网络 =================
@st.cache_resource
def load_ai_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # 1. 检测器
    detector = models.resnet18(weights=None)
    detector.fc = nn.Linear(detector.fc.in_features, 4)
    detector.load_state_dict(torch.load('models/fetal_palate_detector.pth', map_location=device, weights_only=True))
    detector.to(device).eval()

    # 2. 分类器
    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    classifier.load_state_dict(torch.load('models/best_cleft_classifier.pth', map_location=device, weights_only=True))
    classifier.to(device).eval()
    return detector, classifier, device


# ================= 3. 仿生诊断引擎实现 =================
def run_diagnosis_pipeline(original_image, detector, classifier, device):
    orig_w, orig_h = original_image.size

    # 阶段一：目标检测定位靶区 (寻器官)
    det_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    det_input = det_transform(original_image.resize((256, 256))).unsqueeze(0).to(device)
    with torch.no_grad():
        coords = detector(det_input)[0].tolist()

    scale_w, scale_h = orig_w / 256.0, orig_h / 256.0
    xmin, ymin = max(0, coords[0] * scale_w), max(0, coords[1] * scale_h)
    xmax, ymax = min(orig_w, coords[2] * scale_w), min(orig_h, coords[3] * scale_h)

    # 阶段二：靶区特征鉴别 diagnosis (判病理)
    palate_crop = original_image.crop((xmin, ymin, xmax, ymax))
    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cls_input = cls_transform(palate_crop).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classifier(cls_input)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # {'Abnormal_Or_NotPalate': 0, 'NormalPalate': 1}
    prob_normal, prob_cleft = probabilities[1].item(), probabilities[0].item()

    # 阶段三：报告生成 (物理融合)
    draw = ImageDraw.Draw(original_image)
    threshold = 0.60

    if prob_normal > threshold:
        status, color = "NORMAL PALATE", "#00FF00"
        conf = prob_normal
    else:
        status, color = "CLP SUSPECTED", "#FF0000"
        conf = prob_cleft

    # 直接在原图上用医生熟悉的粗线条画出精准靶区
    draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=4)

    return original_image, status, prob_normal, prob_cleft


# ================= 4. 主界面：极简医学看板设计 =================
st.title("🏥 胎儿唇腭裂全自动 AI 诊断系统")
st.markdown("---")  # 分割线

detector, classifier, device = load_ai_models()

# 使用 Streamlit 的容器将上传组件放在一个侧边栏里，主界面留白
uploaded_file = st.sidebar.file_uploader("📂 请选择胎儿超声切面图", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert('RGB')

    # 主界面大图预览 (Apple 风格大面积留白)
    main_container = st.container()

    with main_container:
        st.subheader("影像科AI辅助初筛诊断报告")

        # 核心：将图片和诊断数据分开，给大字留出空间
        report_cols = st.columns([0.6, 0.4], gap="medium")  # 左图右数， gap 防止拥挤

        with report_cols[0]:
            # 用于 st.image 的原图预览
            st.image(original_image, caption="原始影像档案", use_container_width=True)

        with report_cols[1]:
            # 诊断按钮放在诊断数据上方，点击开始诊断
            st.write("\n\n")  # 制造一点间距
            start_diag = st.button("启动智能看片", use_container_width=True)

            if start_diag:
                with st.spinner("双核引擎正在深挖纹理特征..."):
                    processed_image, diagnosis, prob_n, prob_c = run_diagnosis_pipeline(original_image, detector,
                                                                                        classifier, device)

                    # 替换掉左边的原图为带有诊断框的图
                    report_cols[0].empty()  # 清空旧图
                    report_cols[0].image(processed_image, caption="AI 靶区定位与识别报告", use_container_width=True)

                    # ⭐️ 右边使用Metric布局，防止文字溢出⭐️

                    # A. 最终诊断卡片 (大面积背景，强调绿色或红色)
                    if diagnosis == "NORMAL PALATE":
                        st.success(f"🏆 最终诊断: **NORMAL**")
                    else:
                        st.error(f"⚠️ 最终诊断: **CLP SUSPECTED**")

                    st.write("---")  # 小分割线

                    # B. 辅助数据卡片 (使用 metric，自带缩写和适配功能)
                    # 我们的 CSS 已经强制 Metric label 换行，防止出现 ...
                    st.metric(label="正常上腭纹理匹配度", value=f"{prob_n * 100:.1f}%")
                    st.metric(label="疑似唇裂病理特征匹配度", value=f"{prob_c * 100:.1f}%")
                    st.metric(label="诊断置信度", value=f"{max(prob_n, prob_c) * 100:.1f}%", help="基于 softmax 概率")

else:
    # 未上传时的占位符 (防止页面空旷)
    with st.container():
        st.markdown("""
            <div style="padding: 3rem; text-align: center; color: #999; background-color: #f9f9f9; border-radius: 1rem; border: 2px dashed #eee;">
                <h3>👈 请在左侧上传超声图像</h3>
                <p>系统将自动为您裁切解剖结构并出具秒级诊断报告</p>
            </div>
        """, unsafe_allow_html=True)