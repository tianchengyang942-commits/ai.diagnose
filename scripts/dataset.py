import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class FetalFacialDataset(Dataset):
    def __init__(self, annotation_path, img_dir, transform=None, target_size=256):
        self.img_dir = img_dir
        self.transform = transform
        self.target_size = target_size  # 我们统一把图片缩放为 256x256

        # 1. 读取表格
        if annotation_path.endswith('.csv'):
            try:
                df = pd.read_csv(annotation_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(annotation_path, encoding='gbk')
        else:
            df = pd.read_excel(annotation_path)

        df.columns = df.columns.str.strip()

        # 2. 我们只需要腭部 (palate) 的真实坐标作为目标
        df = df[df['structure'] == 'palate'].reset_index(drop=True)

        # 3. 过滤丢失的图片
        valid_rows = []
        for idx, row in df.iterrows():
            img_name = str(row['fname'])
            img_path = os.path.join(self.img_dir, img_name)

            if os.path.exists(img_path):
                valid_rows.append(row)
            else:
                alt_path = img_path.replace('.png', '.jpg')
                if os.path.exists(alt_path):
                    row['fname'] = img_name.replace('.png', '.jpg')
                    valid_rows.append(row)

        self.data_frame = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"🔍 准备就绪：共找到 {len(self.data_frame)} 张包含腭部坐标的有效大图。")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]

        img_name = str(row['fname'])
        img_path = os.path.join(self.img_dir, img_name)

        # 加载完整原图
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size  # 记录原图的宽和高

        # 读取 Excel 里的真实坐标
        # 表格格式：w 对应横向(x)，h 对应纵向(y)
        xmin = float(row['w_min'])
        ymin = float(row['h_min'])
        xmax = float(row['w_max'])
        ymax = float(row['h_max'])

        # ⭐️ 核心逻辑：坐标按比例缩放 ⭐️
        # 因为我们把图片缩放到了 256x256，所以坐标也必须等比例缩小！
        x_scale = self.target_size / orig_w
        y_scale = self.target_size / orig_h

        new_xmin = xmin * x_scale
        new_ymin = ymin * y_scale
        new_xmax = xmax * x_scale
        new_ymax = ymax * y_scale

        # 把原图缩放为 256x256
        image = image.resize((self.target_size, self.target_size))

        if self.transform:
            image = self.transform(image)

        # 返回：1.处理好的完整图片，2.包含 4 个坐标的 Tensor
        bbox = torch.tensor([new_xmin, new_ymin, new_xmax, new_ymax], dtype=torch.float32)

        return image, bbox


# ================= 运行测试模块 =================
if __name__ == "__main__":
    ANNOTATION_PATH = '/Users/tianchengyang/Desktop/Dataset for Fetus Framework/data/ObjectDetection.xlsx'
    IMG_DIR = '/Users/tianchengyang/Desktop/Dataset for Fetus Framework/data/images'

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    try:
        dataset = FetalFacialDataset(annotation_path=ANNOTATION_PATH, img_dir=IMG_DIR, transform=data_transforms)

        if len(dataset) > 0:
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            batch_imgs, batch_bboxes = next(iter(dataloader))
            print(f"🎉 成功！\nAI 看到的图片维度: {batch_imgs.shape}")
            print(f"AI 需要学习的正确坐标答案 (x_min, y_min, x_max, y_max):\n{batch_bboxes}")

    except Exception as e:
        print(f"❌ 运行失败: {e}")