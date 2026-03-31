# Attention 注意力机制详解

> Attention 是 Transformer 的核心，也是本项目 model.py 里最复杂的部分。
> 这一篇把它拆解透。

---

## 1. 为什么需要 Attention？—— 解决"长距离依赖"问题

处理一句话时：

```
短距离依赖（循环神经网络 RNN 能处理）：
  "我喜欢___苹果"  → 填"红"
  （"我"和"喜欢"很近，RNN 能记住）

长距离依赖（RNN 容易遗忘）：
  "今天早上我吃了一颗____，它来自新疆，很甜"  → 填"哈密瓜"
  （开头的信息要传到最后，RNN 在中间会遗忘）
```

Attention 的核心思想：**每个位置都可以直接"关注"序列中的任意其他位置**，不受距离限制。

---

## 2. Attention 的三个角色：Q、K、V

类比数据库查询：

```
你（Query）："我想查一下今天北京的天气"
键（Key）：   数据库里每条记录的标题
值（Value）： 对应的详细内容

Q 和 K 一配对就知道"相关性有多高"
然后用这个相关性对 V 做加权平均，得到答案
```

Attention 数学公式：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
   ↓
1. QK^T        → 算每对 Query-Key 的相似度
2. / √d_k      → 缩放（防止点积过大导致梯度消失）
3. softmax     → 变成概率分布（所有权重和为 1）
4. × V         → 用概率对 Value 加权求和
```

```js
// Attention 的伪代码
function attention(Q, K, V) {
  // 1. QK^T：每个 query 对每个 key 的点积
  const scores = Q.matmul(K.transpose()); // [seq_len, seq_len]

  // 2. 缩放
  const scaled = scores / Math.sqrt(keyDim);

  // 3. softmax：变成概率
  const weights = softmax(scaled, axis=-1); // 每行和为 1

  // 4. 加权求和
  const output = weights.matmul(V);

  return output;
}
```

---

## 3. 多头注意力（Multi-Head Attention）—— 多个"视角"同时看

单一注意力头只能捕捉一种类型的关系。

```js
// 多头注意力：并行跑 H 个独立的 Attention
const heads = [];
for (let h = 0; h < numHeads; h++) {
  const Qh = Q.slice(h); // 第 h 个头的 Q
  const Kh = K.slice(h); // 第 h 个头的 K
  const Vh = V.slice(h); // 第 h 个头的 V
  heads.push(attention(Qh, Kh, Vh)); // 各自独立计算
}
const output = concat(heads); // 拼起来
```

本项目：`n_head=4`，即 4 个头并行计算，每个头关注不同方面的关系：

```
头1：关注 主语-动词 关系（"谁"做了"什么"）
头2：关注 形容词-名词 关系（"什么样的"东西）
头3：关注 上下文相似 关系（同义词/相关词）
头4：关注 位置关系（词在哪里出现）
```

每个头的输出维度 = `n_embd / n_head = 128 / 4 = 32`
最终 4 个头拼接 = `4 × 32 = 128`（恢复原始维度）

---

## 4. 因果掩码（Causal Mask）—— 防止看到"未来"

GPT 是**因果语言模型**（Causal LM），只能根据前文预测下一个词。

```
输入:  [今, 天, 天, 气]
       ↓
我想预测"气"，只能看[今, 天, 天]，
不能偷看"气"本身（那不是作弊吗？）

注意力矩阵应该长这样（True = 能看到）：
                今    天    天    气
          今   ✓     -     -     -
          天   ✓     ✓     -     -
          天   ✓     ✓     ✓     -
          气   ✓     ✓     ✓     ✓

下三角矩阵（包含自身，不包含未来）
```

本项目的 causal mask：

```python
mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
# torch.tril = 取下三角（不含右上角）
# 下三角为 True（能看），右上角为 False（不能看）
```

---

## 5. 本项目的 Attention 实现—— 手把手拆解

```python
# model.py CausalSelfAttention.forward()

# 输入 x：[batch, seq_len, n_embd] = [4, 64, 128]
b, t, c = x.shape

# 1. QKV 投影：一次线性变换得到 Q、K、V
qkv = self.qkv(x)  # [4, 64, 384]（3 × 128 = 384）
q, k, v = qkv.split(c, dim=2)  # 分割成 Q、K、V

# 2. 分头：[4, 64, 128] → [4, 4, 64, 32]
# 4 个头，每头 32 维
q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

# 3. PyTorch 2.0+ 融合版（高效）
# 等价于：softmax(QK^T / √d) × V，但更快（FlashAttention）
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# 4. 恢复形状：[4, 4, 64, 32] → [4, 64, 128]
y = y.transpose(1, 2).contiguous().view(b, t, c)

# 5. 输出投影 + Dropout
y = self.resid_drop(self.proj(y))
```

---

## 6. 什么是"残差连接"？—— 就像 Git 的分支合并

```js
// Express 的中间件链里，有时候会"跳过"一些处理：
function middleware(req, res, next) {
  const result = processA(req); // 处理 A
  req.data = result;
  next(); // 继续往下走
}
```

Transformer 里的**残差连接（Residual Connection）**：

```python
# Block 里的做法：
x = x + self.attn(self.ln_1(x))   # 原始 x + 处理后的 x（维度不变）
x = x + self.mlp(self.ln_2(x))   # 同上
```

为什么需要残差连接？

```
没有残差：深层网络 → 梯度消失 → 训练不动
有残差：    梯度可以直接流过"高速公路"（恒等映射）→ 深层也能训练
```

类比 Git：
```
主分支 (x)
  ↓
开发者分支 (attn(x))  ← 做了修改
  ↓
合并回主分支 (x + attn(x))
  ↓
主分支依然保留了原始信息
```

---

## 7. 为什么需要 LayerNorm？

残差连接之后，紧接着一个 LayerNorm：

```
x = x + self.attn(self.ln_1(x))
         ↓
对 x 做 LayerNorm（标准化）
```

为什么？
- 残差连接让值可能越堆越大（尤其是深层）
- LayerNorm 把每层的数值分布"拉回"稳定范围
- 类似于：你把 10 个不同量纲的数字都除以它们的总和，让它们可比

```js
// LayerNorm 的简化理解
function layerNorm(x) {
  const mean = average(x);      // 算均值
  const std = standardDev(x);   // 算标准差
  return (x - mean) / std;     // 标准化到 N(0,1)
}
```
