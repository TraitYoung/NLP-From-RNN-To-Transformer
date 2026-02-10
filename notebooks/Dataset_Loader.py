import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import glob

class MedicalImageDataset(Dataset):
    """
    毕设专用数据加载器。
    读取文件夹里的图片，并把它们变成 Tensor 喂给模型。
    """
    def __init__(self, img_dir, image_size=64):
        # 1. 找到文件夹里所有的图片 (jpg, png, jpeg)
        self.image_paths = []
        for ext in ['*.jpg', '*.png', '*.jpeg']:
            # 递归查找
            self.image_paths.extend(glob.glob(os.path.join(img_dir, '**', ext), recursive=True))
            
        print(f"🔍 Found {len(self.image_paths)} images in {img_dir}")
        
        # 2. 定义图片预处理 (毕设必须步骤)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), # 统一大小
            transforms.ToTensor(),                       # 变张量 (0~1)
            transforms.Normalize([0.5], [0.5])           # 归一化到 (-1~1)，这是 Diffusion 的标准
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图片
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB") # 强转RGB，防止灰度图报错
            return self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 64, 64) # 坏图返回全黑，防止程序崩溃


# --- 测试代码 ---
if __name__ == "__main__":
    # 陛下，为了测试，请在你电脑的 data 文件夹里随便放几张 jpg 图片！
    # 比如: ../data/demo_images/
    dataset = MedicalImageDataset(img_dir="../data", image_size=64)
    if len(dataset) > 0:
        img = dataset[0]
        print(f"Image Shape: {img.shape}") # 应该是 (3, 64, 64)
        print(f"Value Range: [{img.min():.2f}, {img.max():.2f}]") # 应该是 [-1, 1]
    else:
        print("⚠️ 没找到图片，请在 data 文件夹里放点图测试！")
