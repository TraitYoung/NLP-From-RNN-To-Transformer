import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """
    这是 GPT 的心脏：带掩码的自注意力机制 (Masked Self-Attention)。
    面试必问点：
    1. 为什么要除以 sqrt(head_dim)? -> 防止 Softmax 梯度消失。
    2. 为什么要加 Mask? -> 保证模型只能看见过去，不能看见未来 (自回归属性)。
    """
    def __init__(self, d_model, n_head, max_len=1024):
        super().__init__()
        assert d_model % n_head == 0 # 确保能被整除
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # 定义 Q, K, V 的映射矩阵
        # 相比于分别定义三个 Linear，合并成一个再 split 效率更高
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        
        # 输出投影层
        self.c_proj = nn.Linear(d_model, d_model)
        
        # 定义因果掩码 (Causal Mask)
        # 这是一个下三角矩阵，用来盖住右上角（未来的信息）
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size() # Batch_size, Time_step (Sequence Length), Channels (Embed Dim)
        
        # 1. 计算 Q, K, V
        # qkv shape: (B, T, 3 * C) -> split -> (B, T, C)
        q, k, v = self.c_attn(x).split(C, dim=2)
        
        # 2. 变换形状以适应多头注意力 (Multi-Head)
        # (B, T, n_head, head_dim) -> transpose -> (B, n_head, T, head_dim)
        # 这样 transpose 后，n_head 维度在外，可以并行计算所有头
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. 计算注意力分数 (Scaled Dot-Product Attention)
        # att shape: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 4. 应用因果掩码 (Masking)
        # 将 mask 为 0 的位置填入 -inf，这样 Softmax 后就会变成 0
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        
        # 5. 归一化概率
        att = F.softmax(att, dim=-1)
        
        # 6. 聚合 Value
        y = att @ v # (B, n_head, T, head_dim)
        
        # 7. 拼回原始形状
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)


# ==========================================
# 2. The Bridge: Cross-Attention (For Diffusion / medi-diff)
# ==========================================
class CrossAttention(nn.Module):
    """
    这是 Stable Diffusion / U-Net 的关键组件。
    作用：让图像生成过程"听懂"文本提示词 (Prompt)。
    
    面试必问点：
    Q: Q, K, V 分别来自哪里？
    A: Q 来自图像特征 (Latent Image)，K 和 V 来自文本编码 (CLIP Text Embedding)。
    """
    def __init__(self, d_model, d_context, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        
        # Q 来自图像 (U-Net 的中间层特征)
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        
        # K, V 来自文本 (CLIP 的输出 context) !!!
        # 注意：d_context 通常是 CLIP 的维度 (例如 768)，可能与 d_model 不同
        self.to_k = nn.Linear(d_context, d_model, bias=False)
        self.to_v = nn.Linear(d_context, d_model, bias=False)
        
        self.to_out = nn.Linear(d_model, d_model)

    def forward(self, x, context):
        # x: 图像特征 (Batch, Pixels, Channel) -> 比如 (1, 1024, 320)
        # context: 文本特征 (Batch, Token_Len, Channel) -> 比如 (1, 77, 768)
        
        B, T, C = x.shape
        h = self.n_head
        
        # 1. 计算 Q (图像), K (文本), V (文本)
        q = self.to_q(x).view(B, -1, h, self.head_dim).transpose(1, 2)
        k = self.to_k(context).view(B, -1, h, self.head_dim).transpose(1, 2)
        v = self.to_v(context).view(B, -1, h, self.head_dim).transpose(1, 2)
        
        # 2. 计算注意力分数 (注意：Cross Attention 通常不需要 Causal Mask)
        # 图像的任何一个像素都可以看文本的任何一个词
        dots = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = dots.softmax(dim=-1)
        
        # 3. 聚合信息
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, T, C)
        
        return self.to_out(out)


# --- 联合测试代码 ---
if __name__ == "__main__":
    print("-" * 20)
    print("Testing Self-Attention (GPT mode)...")
    d_model = 64
    n_head = 4
    x = torch.randn(1, 10, d_model)
    block = CausalSelfAttention(d_model=d_model, n_head=n_head)
    output = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Transformer Block forward pass successful!")

    print("-" * 20)
    print("Testing Cross-Attention (Diffusion mode)...")
    unet_dim = 320
    text_dim = 768
    dummy_img = torch.randn(1, 1024, unet_dim)
    dummy_text = torch.randn(1, 77, text_dim)
    cross_block = CrossAttention(d_model=unet_dim, d_context=text_dim, n_head=8)
    output = cross_block(dummy_img, dummy_text)
    print(f"Image Input: {dummy_img.shape}")
    print(f"Text Input:  {dummy_text.shape}")
    print(f"Fused Output:{output.shape}")
    print("Cross-Attention success! Image is now guided by Text.")
