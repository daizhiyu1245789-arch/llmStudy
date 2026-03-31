"""
model.py - GPT 风格因果 Transformer 的核心模型定义
从 0 实现一个极小的语言模型，包含：因果自注意力、MLP、Transformer Block 和完整的 GPT 模型
"""

# ============================================================
# 1. 导入标准库和第三方库
# ============================================================

# 启用"未来"版本特性，让类型注解等新特性在旧版 Python 中可用
from __future__ import annotations

# dataclass 装饰器用于快速创建配置数据类
from dataclasses import dataclass

# Optional 是类型注解的一部分
from typing import Optional

# PyTorch 深度学习框架核心
import torch

# nn 包含所有神经网络层（如 Linear、Embedding、LayerNorm 等）
import torch.nn as nn

# F 包含所有激活函数和损失函数（如 softmax、gelu、cross_entropy 等）
import torch.nn.functional as F


# ============================================================
# 2. GPT 模型配置数据类
# ============================================================

@dataclass
class GPTConfig:
    """
    GPT 模型的超参数配置。

    属性:
        vocab_size: 词表大小。字符级分词时等于字符种类数，字节级分词时固定为 256
        block_size: 上下文窗口长度，即模型一次能处理的最大 token 数
        n_layer: Transformer 编码器的层数（多少个 Block 堆叠）
        n_head: 注意力机制中并行"头"的数量（多头注意力）
        n_embd: 每个 token 嵌入向量的维度（Embedding Dimension）
        dropout: Dropout 比率，用于正则化防止过拟合
        bias: 是否在 LayerNorm 和 Linear 层中使用偏置项（GPT-2 风格为 True）
    """
    vocab_size: int = 256      # 词表大小，默认 256（字节级）
    block_size: int = 128      # 最大上下文长度（位置编码和 causal mask 的尺寸）
    n_layer: int = 4          # Transformer Block 堆叠层数
    n_head: int = 4            # 多头注意力的头数
    n_embd: int = 128          # 嵌入向量维度（每个 token 表示成多长的向量）
    dropout: float = 0.1       # Dropout 概率（训练时随机丢弃的比例）
    bias: bool = True          # 是否使用偏置（LayerNorm 仿射参数、Linear 偏置）


# ============================================================
# 3. 因果自注意力层（Causal Self-Attention）
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    因果自注意力层。

    核心思想：让序列中每个位置只能"看到"当前位置及其之前的内容，
    不能利用未来（右侧）位置的信息。这通过一个下三角的注意力掩码实现。

    实现方式：
    - 将输入分别投影为 Q（查询）、K（键）、V（值）三个向量
    - Q 和 K 做点积得到注意力分数，再通过掩码屏蔽未来位置
    - 用注意力分数对 V 做加权求和，得到上下文表示
    - 最后通过一个线性投影层输出
    """

    def __init__(self, cfg: GPTConfig):
        """
        初始化因果自注意力层。

        参数:
            cfg: GPTConfig 配置对象，包含 n_embd、n_head、block_size 等超参数
        """
        super().__init__()  # 调用 nn.Module 的初始化方法，注册所有子模块

        # 验证 n_embd 必须能被 n_head 整除，以便均匀分配维度
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        # 保存头数和每个头的维度
        self.n_head = cfg.n_head          # 注意力头的数量
        self.head_dim = cfg.n_embd // cfg.n_head  # 每个头的向量维度
        self.dropout = cfg.dropout        # 保存 dropout 概率供前向传播使用

        # -------------------------------------------------------
        # QKV 投影：将输入 x（形状 [b, t, n_embd]）线性变换为
        # 三组向量 Q、K、V（拼接后形状为 [b, t, 3*n_embd]）
        # 这样比分别创建三个线性层更高效
        # -------------------------------------------------------
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)

        # 输出投影：将注意力输出从 [b, t, n_embd] 映射回同一维度
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        # 注意力权重和残差路径上的 Dropout 层
        self.attn_drop = nn.Dropout(cfg.dropout)   # 对注意力概率矩阵做 Dropout
        self.resid_drop = nn.Dropout(cfg.dropout)  # 对最终输出做 Dropout

        # -------------------------------------------------------
        # 注册因果掩码（causal mask）
        # 这是一个下三角矩阵，形状为 [block_size, block_size]
        # 位置 (i, j) = True 表示位置 i 可以关注位置 j
        # 由于 PyTorch 的 torch.tril 默认生成下三角为 1（True），正好符合需求
        # register_buffer 会把 mask 作为模型的持久状态随模型保存/加载，但不参与梯度计算
        # persistent=False 表示不将其写入state_dict（因为可以通过 block_size 重新生成）
        # -------------------------------------------------------
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算给定序列的因果自注意力输出。

        参数:
            x: 输入张量，形状为 [batch_size, seq_len, n_embd]
               batch_size = b, seq_len = t, embed_dim = c

        返回:
            自注意力输出，形状为 [batch_size, seq_len, n_embd]
        """
        b, t, c = x.shape  # 批量大小、序列长度、嵌入维度

        # ---- QKV 投影 ----
        # 将输入 x 线性变换为 Q、K、V 的拼接向量
        qkv = self.qkv(x)  # 形状: [b, t, 3*c]

        # ---- 分割 Q、K、V ----
        # 在最后一维（通道维度 c）上等分为三份，分别得到 Q、K、V
        q, k, v = qkv.split(c, dim=2)  # 每个形状: [b, t, c]

        # ---- 多头并行化 ----
        # 将 Q、K、V 的形状从 [b, t, c] 变为 [b, n_head, t, head_dim]
        # transpose(1, 2) 交换头维和序列维，使得每个头可以独立计算注意力
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # [b, n_head, t, head_dim]
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # [b, n_head, t, head_dim]
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)  # [b, n_head, t, head_dim]

        # -------------------------------------------------------
        # 计算注意力输出
        # PyTorch 2.0+ 提供了融合的 F.scaled_dot_product_attention，
        # 它在底层用 FlashAttention 等算法实现，效率更高且梯度计算更稳定
        # -------------------------------------------------------
        if hasattr(F, "scaled_dot_product_attention"):
            # 融合版本的注意力计算
            # attn_mask=None：使用 is_causal=True 自动生成因果掩码
            # dropout_p：仅在训练时（self.training=True）启用 Dropout
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,               # 不额外传递掩码
                dropout_p=self.dropout if self.training else 0.0,  # 训练/推理时控制 dropout
                is_causal=True,                # 告诉函数生成因果掩码
            )
        else:
            # ---- 手动实现（PyTorch < 2.0 或不支持 FlashAttention 的设备）----
            # 手动计算 Scaled Dot-Product Attention

            # Q @ K^T：计算每个查询对所有键的注意力分数
            # 形状: [b, n_head, t, head_dim] @ [b, n_head, head_dim, t] -> [b, n_head, t, t]
            att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))  # 缩放防止梯度消失

            # 获取当前序列长度对应的因果掩码（取前 t 行 t 列）
            mask = self.causal_mask[:t, :t]  # 形状: [t, t]（布尔张量）

            # 用 -inf 屏蔽未来位置（这些位置的注意力分数被设为一个极大的负数）
            att = att.masked_fill(~mask, float("-inf"))

            # 对最后一维（键的维度）做 softmax，得到注意力概率分布（和为 1）
            att = F.softmax(att, dim=-1)

            # 对注意力概率矩阵应用 Dropout（训练时）
            att = self.attn_drop(att)

            # 用注意力概率对 V 加权求和，得到上下文向量
            y = att @ v  # [b, n_head, t, t] @ [b, n_head, t, head_dim] -> [b, n_head, t, head_dim]

        # ---- 恢复形状 ----
        # 1. transpose(1, 2)：从 [b, n_head, t, head_dim] 变回 [b, t, n_head, head_dim]
        # 2. contiguous()：确保张量在内存中是连续存储的（view 之前必须连续）
        # 3. view(b, t, c)：重新调整为 [b, t, n_embd]（恢复原始形状）
        y = y.transpose(1, 2).contiguous().view(b, t, c)

        # ---- 输出投影 + Dropout ----
        # 通过线性层和 Dropout 得到最终输出
        y = self.resid_drop(self.proj(y))
        return y


# ============================================================
# 4. 多层感知机（MLP / Feed-Forward Network）
# ============================================================

class MLP(nn.Module):
    """
    多层感知机模块，即前馈神经网络（FFN）。

    在 Transformer 中，每个 Block 包含一个注意力层和一个 FFN 层。
    FFN 通常由两个线性变换组成，中间有一个非线性激活函数（这里使用 GELU）。

    结构：n_embd -> 4*n_embd（扩展层）-> n_embd（压缩回原维度）
    扩展系数 4 来自原始 Transformer 论文（也称为"内部维度"）
    """

    def __init__(self, cfg: GPTConfig):
        """
        初始化 MLP 模块。

        参数:
            cfg: GPTConfig 配置对象
        """
        super().__init__()

        # FFN 的隐藏层维度，原始 Transformer 使用 4 倍扩展
        hidden = 4 * cfg.n_embd

        # ---- 第一层线性变换：n_embd -> 4*n_embd ----
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)

        # ---- 第二层线性变换：4*n_embd -> n_embd（投影回原维度）----
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)

        # Dropout 层：在输出上应用正则化
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：输入通过两层线性变换和非线性激活函数。

        参数:
            x: 输入张量，形状 [batch_size, seq_len, n_embd]

        返回:
            输出张量，形状 [batch_size, seq_len, n_embd]
        """
        # 第一层：线性变换 + GELU 激活函数
        # GELU（Gaussian Error Linear Unit）是一种平滑的激活函数，
        # 比 ReLU 更平滑，在 Transformer 中表现更好
        x = self.fc(x)          # [b, t, n_embd] -> [b, t, 4*n_embd]
        x = F.gelu(x)            # 应用 GELU 非线性激活

        # 第二层：线性变换 + Dropout
        x = self.proj(x)        # [b, t, 4*n_embd] -> [b, t, n_embd]
        x = self.drop(x)        # 应用 Dropout 正则化

        return x


# ============================================================
# 5. Transformer Block（一个完整的 Transformer 编码器块）
# ============================================================

class Block(nn.Module):
    """
    Transformer Block（编码器块）。

    每个 Block 包含两个子层：
    1. CausalSelfAttention（因果自注意力）
    2. MLP（多层感知机）

    每个子层外围都有残差连接（Residual Connection）和层归一化（Layer Normalization）。
    用公式表示：
        x = x + SelfAttention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    残差连接让梯度能够直接流过，缓解深层网络的梯度消失问题。
    """

    def __init__(self, cfg: GPTConfig):
        """
        初始化一个 Transformer Block。

        参数:
            cfg: GPTConfig 配置对象
        """
        super().__init__()

        # ---- 第一子层：注意力 + 残差连接 ----
        # Pre-Norm 风格：先做 LayerNorm，再做注意力
        self.ln_1 = nn.LayerNorm(cfg.n_embd, elementwise_affine=True)  # LayerNorm 归一化
        self.attn = CausalSelfAttention(cfg)                            # 因果自注意力

        # ---- 第二子层：MLP + 残差连接 ----
        # Pre-Norm 风格：先做 LayerNorm，再做 MLP
        self.ln_2 = nn.LayerNorm(cfg.n_embd, elementwise_affine=True)  # LayerNorm 归一化
        self.mlp = MLP(cfg)                                             # 多层感知机

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：输入依次通过两个子层（注意力 + MLP），每层都有残差连接。

        参数:
            x: 输入张量，形状 [batch_size, seq_len, n_embd]

        返回:
            输出张量，形状 [batch_size, seq_len, n_embd]
        """
        # ---- 注意力子层 ----
        # 1. 对输入 x 做 LayerNorm
        # 2. 计算自注意力
        # 3. 残差连接：原始输入 + 注意力输出（x 维度不变）
        x = x + self.attn(self.ln_1(x))

        # ---- MLP 子层 ----
        # 1. 对上一步结果做 LayerNorm
        # 2. 通过 MLP 前馈网络
        # 3. 残差连接：输入 + MLP 输出
        x = x + self.mlp(self.ln_2(x))

        return x


# ============================================================
# 6. 完整的 GPT 模型
# ============================================================

class GPT(nn.Module):
    """
    完整的 GPT 模型（GPT-2 风格的因果 Transformer）。

    结构概览：
    - Token Embedding（词表嵌入）：将 token ID 映射为向量
    - Position Embedding（位置嵌入）：为每个位置添加位置信息
    - Dropout
    - N 个 Transformer Block（N = n_layer）
    - 最终的 LayerNorm
    - Language Model Head（LM Head）：将向量映射回词表维度得到 logit

    共享权重技巧：LM Head 和 Token Embedding 共享同一个权重矩阵，
    这是一种常见的节省参数量的做法（tie_weights）。
    """

    def __init__(self, cfg: GPTConfig):
        """
        初始化完整的 GPT 模型。

        参数:
            cfg: GPTConfig 配置对象
        """
        super().__init__()
        self.cfg = cfg  # 保存配置引用

        # -------------------------------------------------------
        # Token Embedding：把 token ID（整数）变成 n_embd 维向量
        # 形状：[vocab_size, n_embd]，即每个 token 有一个专属的嵌入向量
        # vocab_size 行，每行是一个 n_embd 维的向量
        # -------------------------------------------------------
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)

        # -------------------------------------------------------
        # Position Embedding：把位置 ID 变成 n_embd 维向量
        # 形状：[block_size, n_embd]，即每个位置（0 到 block_size-1）有一个嵌入向量
        # 位置编码让模型知道 token 在序列中的位置信息
        # -------------------------------------------------------
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)

        # Dropout 层：在嵌入层输出后应用正则化
        self.drop = nn.Dropout(cfg.dropout)

        # -------------------------------------------------------
        # 堆叠 N 个 Transformer Block
        # nn.ModuleList 将多个 Block 存储为一个列表，PyTorch 会自动注册所有子模块
        # -------------------------------------------------------
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        # 最终的 LayerNorm 层，在所有 Block 之后进行归一化
        self.ln_f = nn.LayerNorm(cfg.n_embd, elementwise_affine=True)

        # -------------------------------------------------------
        # Language Model Head（LM Head）：将隐藏状态映射到词表维度
        # 输出形状：[batch_size, seq_len, vocab_size]，即每个位置的 unnormalized 词表概率（logit）
        # bias=False：通常 LM Head 不需要偏置
        # -------------------------------------------------------
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # ---- 权重共享（tie_weights）----
        # 让 LM Head 的权重和 Token Embedding 的权重相同
        # 这样模型只需要学习一套嵌入矩阵，而不是两套，节省参数量
        # 推理时，LM Head 的权重就是嵌入矩阵的转置
        self.lm_head.weight = self.wte.weight

        # ---- 权重初始化 ----
        # 对所有子模块的参数应用初始化方法
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """
        权重初始化回调函数。

        遍历模型的所有子模块，对特定类型的层应用初始化：
        - Linear：权重用正态分布 N(0, 0.02) 初始化，偏置置零
        - Embedding：权重用正态分布 N(0, 0.02) 初始化

        较小的标准差（0.02）有助于训练初期的稳定性。

        参数:
            module: 模型中的某个子模块
        """
        if isinstance(module, nn.Linear):
            # Linear 层权重：正态分布初始化
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            # 如果有偏置，置为零
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embedding 层权重：同样正态分布初始化
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        完整的前向传播。

        参数:
            idx: 输入 token ID，形状 [batch_size, seq_len]
                 每个元素是 [0, vocab_size-1] 范围内的整数
            targets: 目标 token ID，形状 [batch_size, seq_len]
                     用于计算语言模型损失。如果为 None，则只返回 logits（用于推理）。

        返回:
            logits: 预测 logits，形状 [batch_size, seq_len, vocab_size]
            loss: 交叉熵损失（仅当 targets 不为 None 时计算）
        """
        b, t = idx.shape  # 批量大小、序列长度

        # ---- 序列长度检查 ----
        # 确保输入序列不超过模型支持的最大长度
        if t > self.cfg.block_size:
            raise ValueError(
                f"输入序列长度 {t} 超过了模型支持的最大长度 {self.cfg.block_size}。"
                "请减小输入长度或增大 block_size 配置。"
            )

        # -------------------------------------------------------
        # 构建输入嵌入：Token Embedding + Position Embedding
        # -------------------------------------------------------

        # 位置索引：[0, 1, 2, ..., t-1]，形状 [t]
        pos = torch.arange(0, t, device=idx.device, dtype=torch.long)

        # Token 嵌入：idx 中的每个 token ID 映射为一个 n_embd 维向量
        # idx 形状 [b, t] -> wte(idx) 形状 [b, t, n_embd]
        token_emb = self.wte(idx)

        # 位置嵌入：每个位置映射为一个 n_embd 维向量
        # pos 形状 [t] -> wpe(pos) 形状 [t, n_embd]
        # [None, :, :] 将其广播到 [1, t, n_embd] 以便和 token_emb 相加
        pos_emb = self.wpe(pos)[None, :, :]

        # 两者相加：得到结合了 token 语义和位置信息的嵌入向量
        x = token_emb + pos_emb

        # 应用 Dropout（训练时随机丢弃部分神经元）
        x = self.drop(x)

        # -------------------------------------------------------
        # 通过所有 Transformer Block
        # -------------------------------------------------------
        for blk in self.blocks:
            x = blk(x)  # 每层保持形状 [b, t, n_embd]

        # 最终 LayerNorm 归一化
        x = self.ln_f(x)

        # -------------------------------------------------------
        # LM Head：将隐藏状态映射为词表维度的 logit
        # -------------------------------------------------------
        logits = self.lm_head(x)  # 形状 [b, t, vocab_size]

        # ---- 计算损失（仅在提供 targets 时）----
        loss = None
        if targets is not None:
            # 将 logits 和 targets 展平为 1D 张量
            # logits.view(-1, vocab_size)：将 [b*t, vocab_size]
            # targets.view(-1)：将 [b*t]
            # F.cross_entropy 会自动做 softmax + 负对数似然损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # [b*t, vocab_size]
                targets.view(-1)                  # [b*t]
            )

        return logits, loss

    @torch.no_grad()  # 推理时不需要梯度计算，加速并节省显存
    def generate(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成：基于给定前缀逐步生成后续 token。

        过程：每次让模型基于当前整个序列（已生成+原始前缀）预测下一个 token，
        将该 token 追加到序列末尾，重复直到生成足够数量的 token。

        参数:
            idx: 输入序列，形状 [batch_size, seq_len]，包含已知的 token ID
            max_new_tokens: 需要生成的新 token 数量
            temperature: 温度参数，控制随机性。
                - T=1.0：保持原始 softmax 分布
                - T>1.0：分布更平坦，增加随机性（更"有创造力"）
                - T<1.0：分布更尖锐，增加确定性（更"保守"）
                - T<=0：贪婪选择，直接取概率最大的 token
            top_k: 如果指定，则只保留概率最高的 k 个 token，其余设为 -inf

        返回:
            包含原始前缀和生成 token 的完整序列，形状 [batch_size, seq_len + max_new_tokens]
        """
        self.eval()  # 确保模型处于评估模式（关闭 Dropout）

        # ---- 自回归生成循环 ----
        for _ in range(max_new_tokens):
            # 如果序列长度超过 block_size，截断到最近 block_size 个 token
            # 这是因为位置编码只能支持到 block_size 的长度
            idx_cond = idx[:, -self.cfg.block_size:]

            # 前向传播：基于当前序列预测下一个 token
            # 返回的 loss 忽略（不需要）
            logits, _ = self(idx_cond)

            # 只取最后一个位置的 logit（预测下一个 token）
            # 形状从 [batch_size, seq_len, vocab_size] -> [batch_size, vocab_size]
            logits = logits[:, -1, :]

            # ---- 温度采样 ----
            if temperature <= 0:
                # temperature <= 0：贪婪解码，直接选择概率最大的 token
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat([idx, next_id], dim=1)
                continue

            # 应用温度缩放：logits = logits / temperature
            # 除以温度后，大的值变得更大（更尖锐），小的值变得更小（更平坦）
            logits = logits / temperature

            # ---- Top-K 过滤 ----
            if top_k is not None:
                # 取概率最高的 top_k 个值
                v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                # 取这 top_k 中最小值作为截断阈值
                cutoff = v[:, -1].unsqueeze(-1)  # 形状 [batch_size, 1]
                # 将低于阈值的 logits 设为 -inf（概率为 0）
                logits = torch.where(
                    logits < cutoff,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            # ---- 从分布中采样 ----
            # 将 logits 转为概率分布，然后按概率随机采样一个 token
            probs = F.softmax(logits, dim=-1)  # 形状 [batch_size, vocab_size]
            next_id = torch.multinomial(probs, num_samples=1)  # 形状 [batch_size, 1]
            idx = torch.cat([idx, next_id], dim=1)  # 追加到序列末尾

        return idx
