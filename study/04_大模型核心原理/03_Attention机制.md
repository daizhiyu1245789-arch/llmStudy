# Attention 机制详解

> Attention 是 Transformer 的核心，也是本项目最复杂的部分。

---

## 1. Attention 要解决什么问题？

RNN 处理长序列时，早期的信息会"被遗忘"：

```
RNN 处理："今天早上我吃了一颗____，它来自新疆"
问题：预测"哈密瓜"需要记住"新疆"这个关键信息
RNN 的问题：中间步骤太多，早期信息传不过来
```

Attention 的解决方案：**每个位置可以直接关注序列中的任意其他位置**，不受距离限制。

---

## 2. Attention 的三个主角：Q、K、V

```
Query（Q）= 我当前要查询的内容
Key（K）= 我拥有的信息索引
Value（V）= 索引对应的内容
```

类比图书馆查询：

```
你（Q）: "我想查天气相关的书"
索引（K）: ["烹饪", "旅游", "科技", "天气"]  # 书名
内容（V）: [书A, 书B, 书C, 书D]              # 实际内容

Q 和 K 匹配：
"天气" 和 "烹饪" → 低相关
"天气" 和 "旅游" → 中等相关
"天气" 和 "天气" → 高相关

→ 应该多看"旅游"和"天气"分类的书
```

---

## 3. Attention 的数学公式

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

```
Step 1: QK^T       → 计算 Query 和 Key 的相似度矩阵
Step 2: / √d_k     → 缩放，防止点积过大导致 softmax 梯度消失
Step 3: softmax    → 变成概率分布（每行和为1）
Step 4: × V        → 用概率对 Value 加权求和
```

```python
# 手写 Attention
def attention(Q, K, V, d_k):
    scores = Q @ K.transpose(-2, -1)   # [seq, seq]
    scores = scores / (d_k ** 0.5)     # 缩放
    weights = F.softmax(scores, dim=-1) # [seq, seq]，每行和为1
    output = weights @ V              # [seq, d_model]
    return output
```

---

## 4. 多头注意力（Multi-Head Attention）

单一注意力头只能捕捉一种关系。多个头并行工作，捕捉不同类型的关系。

```python
# 4 个头并行计算（本项目的配置）
n_head = 4
head_dim = n_embd // n_head  # 128 / 4 = 32

# 每个头独立计算 Attention
for h in range(n_head):
    Q_h = Q[:, :, h*head_dim:(h+1)*head_dim]  # 取第 h 个头的 Q
    K_h = K[:, :, h*head_dim:(h+1)*head_dim]
    V_h = V[:, :, h*head_dim:(h+1)*head_dim]
    head_h = attention(Q_h, K_h, V_h, head_dim)

# 4 个头的输出拼接
output = concat([head_0, head_1, head_2, head_3])  # [batch, seq, 128]
```

本项目 4 个头可能分别关注：
```
头1：主语-动词关系（谁做了什么）
头2：形容词-名词关系（什么样的）
头3：上下文相似关系（同义词）
头4：位置关系（词在哪出现）
```

---

## 5. 因果掩码（Causal Mask）

GPT 是因果模型，只能看前文，不能偷看后面的内容。

```python
# 本项目的 causal mask：下三角矩阵
mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))

# mask 结果：
#              看谁→
#       今    天    天    气
# 看  今  True  False False False
# 哪  天  True  True  False False
# 些  天  True  True  True  False
# 人  气  True  True  True  True
```

```python
# 应用掩码
att = att.masked_fill(~mask, float("-inf"))
# 被 mask 的位置（右上角）变成 -inf，softmax 后概率为 0
```

---

## 6. 本项目的 Attention 代码逐行解析

```python
# model.py CausalSelfAttention.forward()

b, t, c = x.shape  # batch=4, seq_len=64, n_embd=128

# 1. QKV 投影
qkv = self.qkv(x)  # [4, 64, 384]，一次线性变换得到 Q,K,V
q, k, v = qkv.split(c, dim=2)  # 分割成三份

# 2. 分头
q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
# [4, 64, 128] → [4, 64, 4, 32] → [4, 4, 64, 32]
# 4个样本，4个头，每头序列长64，每头维度32

# 3. PyTorch 2.0+ 融合版（高效，用 FlashAttention）
y = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=self.dropout if self.training else 0.0,
    is_causal=True,  # 自动生成因果掩码
)

# 4. 恢复形状
y = y.transpose(1, 2).contiguous().view(b, t, c)
# [4, 4, 64, 32] → [4, 64, 4, 32] → [4, 64, 128]

# 5. 输出投影
y = self.resid_drop(self.proj(y))
```

---

## 7. 残差连接（Residual Connection）

残差 = 抄近道，让梯度直接流过。

```python
# 每个 Block 里的两个残差连接
x = x + self.attn(self.ln_1(x))   # 注意力残差
x = x + self.mlp(self.ln_2(x))    # MLP 残差
```

```
没有残差：x → Block(x) → 输出（梯度要穿过整个 Block）
有残差：  x → Block(x) → x + Block(x) = 输出
         梯度可以直接通过 x 直接传回去（恒等梯度）
```

**类比**：就像 Git 的分支合并，主分支（x）和特性分支（Block(x)）合并，保留了原始信息。

---

## 8. Pre-Norm vs Post-Norm

本项目用的是 **Pre-Norm**（先 LayerNorm 再子层）：

```python
# Pre-Norm（本项目用法）
x = x + attn(ln_1(x))    # 先 Norm，再 Attention
x = x + mlp(ln_2(x))      # 先 Norm，再 MLP

# Post-Norm（传统 Transformer 用法）
x = ln(x + attn(x))       # 先算 Attention，再 Norm
```

Pre-Norm 更适合训练深层网络（更稳定）。

---

## 9. 完整 Block 的数据流

```
输入 x: [batch, seq, n_embd]
  ↓
LayerNorm
  ↓
QKV 投影
  ↓
分头 + 计算 Attention（含 causal mask）
  ↓
拼接 + 输出投影
  ↓
Dropout
  ↓
残差连接 x = x + attn_out
  ↓
LayerNorm
  ↓
MLP（FFN：扩展→激活→压缩）
  ↓
Dropout
  ↓
残差连接 x = x + mlp_out
  ↓
输出: [batch, seq, n_embd]（维度不变）
```
