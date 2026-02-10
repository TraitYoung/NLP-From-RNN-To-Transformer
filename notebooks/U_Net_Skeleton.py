import torch
import torch.nn as nn
from Transformer_Core import CrossAttention 

class ResNetBlock(nn.Module):
    """
    基础的卷积模块，用于处理图像特征。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU() # Diffusion 常用激活函数
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        h = self.act(self.conv1(x))
        return self.conv2(h) + x # 残差连接

class SpatialTransformer(nn.Module):
    """
    【关键模块】：将文本信息注入图像的地方。
    """
    def __init__(self, channels, context_dim):
        super().__init__()
        # GroupNorm 需要通道数能被 32 整除，所以 channels 至少要是 32，通常是 64, 128, 320...
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)
        self.transformer = CrossAttention(d_model=channels, d_context=context_dim, n_head=4)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, context):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = x.flatten(2).transpose(1, 2) 
        x = self.transformer(x, context)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return self.proj_out(x) + x_in

class SimpleUNet(nn.Module):
    """
    极简版 U-Net 架构 (已修复通道数问题)
    """
    def __init__(self, in_channels=3, model_channels=64, context_dim=768):
        super().__init__()
        
        # 1. 【新增】入口层：把 3 通道图片变成 64 通道特征
        # 这解决了 "3 不能被 32 整除" 的报错
        self.init_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # 这里的通道数全部改成 model_channels (64)
        self.down_blocks = nn.ModuleList([
            ResNetBlock(model_channels, model_channels),
            ResNetBlock(model_channels, model_channels)
        ])
        
        # 中间层
        self.mid_block1 = ResNetBlock(model_channels, model_channels)
        self.mid_attn = SpatialTransformer(model_channels, context_dim) 
        self.mid_block2 = ResNetBlock(model_channels, model_channels)
        
        # 上采样
        self.up_blocks = nn.ModuleList([
            ResNetBlock(model_channels, model_channels),
            ResNetBlock(model_channels, model_channels)
        ])

        # 【新增】出口层：把 64 通道特征变回 3 通道图片 (为了预测噪声)
        self.out_conv = nn.Conv2d(model_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, context):
        # x: (Batch, 3, 64, 64)
        
        # 先升维：(Batch, 64, 64, 64)
        x = self.init_conv(x)
        
        # Down
        for block in self.down_blocks:
            x = block(x)
            
        # Middle
        x = self.mid_block1(x)
        x = self.mid_attn(x, context)
        x = self.mid_block2(x)
        
        # Up
        for block in self.up_blocks:
            x = block(x)
            
        # 后降维：(Batch, 3, 64, 64)
        return self.out_conv(x)