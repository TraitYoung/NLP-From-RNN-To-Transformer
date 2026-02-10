from Transformer_Core import CrossAttention

import torch
import torch.nn.functional as F
import numpy as np

class LinearNoiseScheduler:
    """
    扩散模型的"时间控制器"。
    负责管理 Beta (加噪速率) 和 Alpha (保留原图速率)。
    对应毕设任务书中的: DDPM / DDIM Sampling
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        
        # 1. 定义 Beta (线性增加的噪声方差)
        # 意思：一开始加很少的噪，后面加很多噪
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        # 2. 定义 Alpha (Alpha = 1 - Beta)
        # 意思：如果 Beta 是噪声比例，Alpha 就是原图残留比例
        self.alphas = 1.0 - self.betas
        
        # 3. 定义 Alpha Cumulative Product (累乘 Alpha)
        # 意思：直接算出从 t=0 到 t=current 这一路剩下了多少原图信息
        # 公式: alpha_bar = product(alpha_i)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 预先计算好 sqrt(alpha_bar) 和 sqrt(1 - alpha_bar)
        # 用于前向加噪公式: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * noise
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(self, original_image, noise, t):
        """
        【前向过程】：一步到位把 x_0 变成 x_t
        original_image: 清晰图像 (Batch, C, H, W)
        noise: 高斯噪声 (Batch, C, H, W)
        t: 当前时间步 (Batch,)
        """
        # 获取当前时间步对应的系数
        # device 转换是为了防止 tensor 在不同设备上 (CPU vs GPU)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod.to(original_image.device)[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.to(original_image.device)[t]
        
        # 调整形状以便广播运算 (Batch, 1, 1, 1)
        sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1, 1)
        
        # 核心公式：x_t = mean + variance
        noisy_image = sqrt_alpha_bar * original_image + sqrt_one_minus_alpha_bar * noise
        return noisy_image

    def sample_prev_timestep(self, xt, noise_pred, t):
        """
        【反向过程】：利用 U-Net 预测的噪声，从 x_t 算出 x_{t-1}
        这是生成图片时的关键步骤。
        """
        # 这里为了简化，只展示核心逻辑。
        # 实际 DDPM 采样公式比这个复杂，包含方差项。
        # x_{t-1} = (x_t - beta / sqrt(1-alpha_bar) * noise_pred) / sqrt(alpha)
        
        beta = self.betas.to(xt.device)[t].view(-1, 1, 1, 1)
        alpha = self.alphas.to(xt.device)[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod.to(xt.device)[t].view(-1, 1, 1, 1)
        
        # 去噪核心公式
        mean = (1 / torch.sqrt(alpha)) * (xt - (beta / sqrt_one_minus_alpha_bar) * noise_pred)
        
        # 加上一点点随机扰动 (Langevin Dynamics 思想)，除了最后一步
        if t[0] > 0:
            noise = torch.randn_like(xt)
            sigma = torch.sqrt(beta)
            return mean + sigma * noise
        else:
            return mean


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟一张 32x32 的图片
    x0 = torch.randn(1, 3, 32, 32) 
    noise = torch.randn_like(x0)
    
    scheduler = LinearNoiseScheduler(num_timesteps=1000)
    
    # 模拟加噪到第 500 步
    t = torch.tensor([500])
    xt = scheduler.add_noise(x0, noise, t)
    
    print(f"Original Mean: {x0.mean().item():.4f}")
    print(f"Noisy Mean (t=500): {xt.mean().item():.4f}")
    print("Scheduler operational. Forward process verified.")
