import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# --- 导入我们之前写的模块 (自家军火库) ---
# 必须确保文件名完全一致，否则报错
from U_Net_Skeleton import SimpleUNet        # 对应 07_UNet_Skeleton.py (请重命名或修改import)
from Diffusion_Scheduler import LinearNoiseScheduler # 对应 08_Diffusion_Scheduler.py
from Dataset_Loader import MedicalImageDataset # 对应 09_Dataset_Loader.py

# --- 配置参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 # 显存小就改小
LR = 1e-4
EPOCHS = 100 # 跑几轮

def train():
    print(f"🚀 Training on {DEVICE}...")
    
    # 1. 准备模型、调度器、数据
    model = SimpleUNet(in_channels=3, context_dim=768).to(DEVICE) # 图片是3通道(RGB)
    scheduler = LinearNoiseScheduler()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # 均方误差：预测噪声 vs 真实噪声

    # --- 自动定位路径 ---
    # 1. 获取当前脚本 (Train_Simple.py) 所在的绝对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 算出项目根目录 (notebooks 的上一级)
    project_root = os.path.dirname(current_script_dir)

    # 3. 拼出 data 文件夹的绝对路径
    data_path = os.path.join(project_root, 'data')

    print(f"📍 Script Location: {current_script_dir}")
    print(f"📂 Looking for Data in: {data_path}")

    # --- 加载数据 ---
    dataset = MedicalImageDataset(img_dir=data_path, image_size=32)
    if len(dataset) == 0:
        print("❌ 没数据跑不了！请在 data 文件夹里放几张图。")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(DEVICE)
            curr_batch = images.shape[0]
            
            # A. 随机采样时间步 t (比如第 50 步，第 900 步...)
            t = torch.randint(0, 1000, (curr_batch,)).to(DEVICE)
            
            # B. 生成随机噪声 noise (这就是我们要预测的目标)
            noise = torch.randn_like(images).to(DEVICE)
            
            # C. 加噪 (Forward Process): x_t = scheduler(x_0, t, noise)
            # 注意：这里调用了我们昨天写的数学公式！
            noisy_images = scheduler.add_noise(images, noise, t)
            
            # D. 假装有一个文本提示 (实际毕设要用 CLIP 编码，这里先用随机向量代替)
            # 形状: (Batch, 77, 768)
            dummy_text_context = torch.randn(curr_batch, 77, 768).to(DEVICE)
            
            # E. 模型预测噪声 (Reverse Process)
            predicted_noise = model(noisy_images, dummy_text_context)
            
            # F. 算 Loss & 反向传播
            loss = criterion(predicted_noise, noise) # 猜的噪声 vs 真的噪声
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # 打印进度
        avg_loss = total_loss / len(dataloader)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.5f}")
            
    print("🎉 训练完成！模型已学会如何去噪。")
    # torch.save(model.state_dict(), "medi_diff_v1.pth")

if __name__ == "__main__":
    train()
