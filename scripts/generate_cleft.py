import os
import pandas as pd
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ================= 1. 配置路径与参数 =================
ANNOTATION_PATH = '/Users/tianchengyang/Desktop/Dataset for Fetus Framework/data/ObjectDetection.xlsx'
IMG_DIR = '/Users/tianchengyang/Desktop/Dataset for Fetus Framework/data/images'

# 终极版鲁棒数据集
OUTPUT_DATASET_DIR = '/Users/tianchengyang/Desktop/Synthetic_Robust_CLP_Dataset'
# ImageFolder 字典排序，确保 Cleft(0) 排在 Normal(1) 前面
CLASS_0_DIR = os.path.join(OUTPUT_DATASET_DIR, 'Class_0_Abnormal_Or_NotPalate')
CLASS_1_DIR = os.path.join(OUTPUT_DATASET_DIR, 'Class_1_NormalPalate')

os.makedirs(CLASS_0_DIR, exist_ok=True)
os.makedirs(CLASS_1_DIR, exist_ok=True)

# 处理 excel 路径容错
if not os.path.exists(ANNOTATION_PATH):
    csv_alt = ANNOTATION_PATH.replace('.xlsx', '.xlsx - ObjectDetection.csv')
    if os.path.exists(csv_alt):
        ANNOTATION_PATH = csv_alt


# ================= 2. 物理声学唇裂合成引擎 (保持逼真) =================
def synthesize_realistic_cleft(image):
    img_np = np.array(image).astype(np.float32)
    h, w, c = img_np.shape
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    center_x = random.randint(int(w * 0.35), int(w * 0.65))
    center_y = random.randint(int(h * 0.3), int(h * 0.6))
    gap_width = random.randint(5, 12)
    gap_length = random.randint(15, 30)
    points = []
    for i in range(6):
        points.append((center_x + random.randint(-gap_width, gap_width),
                       center_y + (i - 3) * (gap_length / 6.0) + random.randint(-4, 4)))
    draw.polygon(points, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3))
    shadow_mask = Image.new('L', (w, h), 0)
    shadow_draw = ImageDraw.Draw(shadow_mask)
    shadow_points = [(center_x - gap_width, center_y), (center_x + gap_width, center_y), (center_x + gap_width + 15, h),
                     (center_x - gap_width - 15, h)]
    shadow_draw.polygon(shadow_points, fill=120)
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=8))
    mask_np = np.array(mask).astype(np.float32) / 255.0
    mask_np = np.expand_dims(mask_np, axis=2)
    shadow_np = np.array(shadow_mask).astype(np.float32) / 255.0
    shadow_np = np.expand_dims(shadow_np, axis=2)
    fluid_texture = np.ones_like(img_np) * random.randint(15, 35) + np.random.normal(0, 8, img_np.shape)
    img_np = img_np * (1 - shadow_np * 0.4)
    img_np = img_np * (1 - mask_np) + fluid_texture * mask_np
    return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


# ================= 3. 开始构建鲁棒数据集 =================
if __name__ == "__main__":
    print("🧬 正在启动 [医学鲁棒版] 超声数据集合成引擎...")
    df = pd.read_excel(ANNOTATION_PATH) if ANNOTATION_PATH.endswith('.xlsx') else pd.read_csv(ANNOTATION_PATH)
    df.columns = df.columns.str.strip()

    # 3.1 处理阳性/健康组 (Palate)
    palate_df = df[df['structure'] == 'palate'].reset_index(drop=True)
    print(f"✅ 找到正常腭部记录: {len(palate_df)} 条，开始生成...")

    count = 0
    for idx, row in palate_df.iterrows():
        img_name = str(row['fname'])
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path): img_path = img_path.replace('.png', '.jpg')
        if not os.path.exists(img_path): continue

        orig_img = Image.open(img_path).convert('RGB')

        # 严格裁剪靶区 (不向外扩，只留腭部)
        target_crop = orig_img.crop(
            (float(row['w_min']), float(row['h_min']), float(row['w_max']), float(row['h_max'])))
        target_crop = target_crop.resize((128, 128))  # 训练小模型 128x128 足够

        # A. 存入正常组 (Class 1)
        target_crop.save(os.path.join(CLASS_1_DIR, f"normal_{img_name}"))

        # B. 生成逼真唇裂并存入非正常组 (Class 0)
        cleft_img = synthesize_realistic_cleft(target_crop)
        cleft_img.save(os.path.join(CLASS_0_DIR, f"cleft_{img_name}"))
        count += 1

    generated_cleft_count = count
    print(f"✅ Class 1 (正常) 生成完毕: {generated_cleft_count}张。")

    # 3.2 ⭐️ [核心修正] 处理干扰组 (非 Palate 器官) ⭐️
    print(f"🔍 正在从表格提取其他器官干扰样本，用于增强 Class 0...")

    # 筛选出所有不是腭部的结构 (丘脑、鼻骨等)
    other_structures_df = df[df['structure'] != 'palate'].reset_index(drop=True)

    # 我们只需要数量和 Class 1 差不多的干扰项，随机挑选 800 条
    num_to_sample = min(len(other_structures_df), generated_cleft_count + 100)
    sampled_others = other_structures_df.sample(n=num_to_sample)

    other_count = 0
    for idx, row in sampled_others.iterrows():
        img_name = str(row['fname'])
        img_path = os.path.join(IMG_DIR, img_name)
        if not os.path.exists(img_path): img_path = img_path.replace('.png', '.jpg')
        if not os.path.exists(img_path): continue

        try:
            orig_img = Image.open(img_path).convert('RGB')
            # 裁剪其他器官的靶区 (它们看起来和腭部纹理完全不同)
            other_crop = orig_img.crop(
                (float(row['w_min']), float(row['h_min']), float(row['w_max']), float(row['h_max'])))
            other_crop = other_crop.resize((128, 128))

            # 存入 Class 0 (非正常腭部组)
            structure_tag = str(row['structure']).replace(' ', '_')
            other_crop.save(os.path.join(CLASS_0_DIR, f"other_{structure_tag}_{other_count}_{img_name}"))
            other_count += 1
        except Exception:
            continue

    print(f"\n🎉 鲁棒数据集构建完毕！")
    print(f"👉 正常腭部 (Class 1): {generated_cleft_count} 张。")
    print(f"👉 非正常/其他 (Class 0): {generated_cleft_count}(合成唇裂) + {other_count}(真实其他器官) 张。")
    print(f"存放在桌面: {OUTPUT_DATASET_DIR}，请重新运行 train_classifier.py！")