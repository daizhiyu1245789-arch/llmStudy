# PyTorch 快速入门

> PyTorch = 深度学习版的 React。学会了 React 的组件和状态，PyTorch 就很好理解。

---

## 1. 核心类比：React ↔ PyTorch

| React | PyTorch | 说明 |
|--------|---------|------|
| `class MyComponent extends React.Component` | `class MyModule(nn.Module)` | 定义组件/模型 |
| `this.state = { count: 0 }` | `self.linear = nn.Linear(128, 256)` | 存放可学习的状态 |
| `render()` | `forward()` | 组件的核心渲染逻辑 |
| JSX | PyTorch 张量 | 数据结构 |
| props | 输入张量 | 组件的输入 |
| `setState({...})` | `optimizer.step()` | 更新状态 |

---

## 2. 张量（Tensor）= 升级版 NumPy 数组

```python
import torch
import numpy as np

# 从列表创建
x = torch.tensor([1, 2, 3])
print(x)       # tensor([1, 2, 3])

# 常用创建方式
torch.zeros(3, 4)       # 3行4列全零矩阵
torch.ones(2, 3)        # 全一矩阵
torch.randn(4, 5)       # 标准正态分布随机（均值为0，方差为1）
torch.rand(3, 4)        # [0, 1)均匀分布随机
torch.arange(0, 10, 2)   # [0, 2, 4, 6, 8]（类似 range）

# NumPy 互转
np_arr = np.array([1, 2, 3])
torch_arr = torch.from_numpy(np_arr)  # numpy → torch
np_back = torch_arr.numpy()           # torch → numpy（共享内存，改一个另一个也变）

# 查看形状
print(x.shape)   # torch.Size([3])
print(x.shape[0]) # 3
```

---

## 3. 张量的形状（Shape）

形状 = 维度的大小，类似 NumPy 的 shape。

```python
# 标量（0维）
scalar = torch.tensor(5)
print(scalar.shape)  # torch.Size([])

# 向量（1维）
vec = torch.randn(10)
print(vec.shape)     # torch.Size([10])

# 矩阵（2维）
mat = torch.randn(4, 8)
print(mat.shape)     # torch.Size([4, 8])

# 3维（批次序列特征）
batch_seq_feat = torch.randn(32, 64, 128)
# 32个样本，序列长度64，每个token 128维向量
print(batch_seq_feat.shape)  # torch.Size([32, 64, 128])

# 本项目 model.py 里的张量形状：
# x: [batch_size, seq_len, n_embd] = [4, 128, 128]
```

---

## 4. 张量运算

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 基本运算
c = a + b        # [5, 7, 9]
c = a - b        # [-3, -3, -3]
c = a * b        # [4, 10, 18] 逐元素乘法（不是矩阵乘法）
c = a / b        # [0.25, 0.4, 0.5]
c = a ** 2       # [1, 4, 9] 平方
c = torch.sqrt(a)  # [1, 1.414, 1.732] 开方

# 矩阵乘法（@ 是 Python 3.5+ 的运算符）
# [m, n] @ [n, k] = [m, k]
x = torch.randn(3, 4)
y = torch.randn(4, 5)
z = x @ y           # torch.Size([3, 5])

# 等价于
z = torch.matmul(x, y)
```

---

## 5. 形状变换

```python
x = torch.randn(4, 8, 128)   # [4, 8, 128]

# view = NumPy 的 reshape（不改变内存布局）
x_flat = x.view(4, -1)      # [4, 1024]，-1表示自动推断
x_again = x_flat.view(4, 8, 128)  # 恢复形状

# transpose = 交换维度
x_t = x.transpose(0, 2)      # [128, 8, 4]

# permute = 任意维度重排
x_p = x.permute(2, 0, 1)     # [128, 4, 8]

# contiguous = 确保内存连续（view 之前可能需要）
y = x.transpose(0, 2)
y = y.contiguous()           # view 之前需要先 contiguous

# squeeze / unsqueeze = 删除/添加维度为1的轴
y = x.unsqueeze(0)          # [1, 4, 8, 128]
y = y.squeeze(0)            # [4, 8, 128]，去掉第0维（如果是1的话）
y = y.squeeze()             # 去掉所有维度为1的轴
```

---

## 6. nn.Module = 定义模型的基类

```python
import torch.nn as nn

# 定义一个模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()   # 必写！调用父类初始化
        self.linear1 = nn.Linear(128, 256)  # 可学习层
        self.relu = nn.ReLU()                # 激活函数
        self.linear2 = nn.Linear(256, 10)    # 输出层

    def forward(self, x):
        x = self.linear1(x)   # 线性变换
        x = self.relu(x)      # 非线性激活
        x = self.linear2(x)
        return x

# 创建模型实例
model = SimpleModel()
print(model)  # 打印模型结构

# 前向传播（直接调用 model()，会自动调用 forward）
input_tensor = torch.randn(32, 128)   # 32个样本，每个128维
output = model(input_tensor)          # 自动调用 forward
print(output.shape)                   # torch.Size([32, 10])
```

---

## 7. nn.Module 的常用层

```python
# 线性层（全连接层）
linear = nn.Linear(in_features=128, out_features=256, bias=True)
# 输入: [batch, 128] → 输出: [batch, 256]

# Embedding 层（词表嵌入）
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128)
# 输入: [batch, seq_len] 的 token ID → 输出: [batch, seq_len, 128]
# 10000 = 词表大小，128 = 每个词的向量维度

# LayerNorm（层归一化）
ln = nn.LayerNorm(128)
# 输入: [batch, seq_len, 128] → 输出: [batch, seq_len, 128]（形状不变）

# Dropout（正则化）
dropout = nn.Dropout(p=0.1)  # 训练时随机丢弃10%
x = dropout(x)               # 推理时自动关闭

# Sequential（顺序容器）
net = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
output = net(input_tensor)
```

---

## 8. nn.functional（F 模块）

`F` 包含不需要保存状态的函数（无参数）。

```python
import torch.nn.functional as F

# 常用激活函数
F.relu(x)         # ReLU: max(0, x)
F.gelu(x)         # GELU: 平滑版 ReLU（本项目用的）
F.sigmoid(x)      # Sigmoid: 1/(1+e^-x)
F.softmax(x, dim=-1)  # Softmax（dim 指定在哪个维度做）

# 损失函数
F.cross_entropy(logits, targets)  # 交叉熵损失（本项目用的）
F.mse_loss(pred, target)          # 均方误差损失

# 其他
F.dropout(x, p=0.1, training=True)  # Dropout
F.layer_norm(x, [128])              # LayerNorm
```

---

## 9. 自动求导（autograd）

PyTorch 的自动求导系统会自动计算梯度。

```python
# requires_grad=True = 标记这个张量需要梯度
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 对 x 做运算，构建计算图
y = x ** 2            # y = x^2
z = y.sum()           # z = sum(y) = x1^2 + x2^2 + x3^2

# 反向传播：自动计算 dz/dx
z.backward()

# 查看梯度
print(x.grad)         # tensor([2., 4., 6.])  ← dz/dx = 2x
# x.grad = [2*1, 2*2, 2*3] = [2, 4, 6]

# 本项目的训练循环里：
loss.backward()        # 自动计算所有参数的梯度
optimizer.step()       # 用梯度更新参数
optimizer.zero_grad()  # 清零梯度，准备下一个 step
```

**计算图** = 记录每一步运算的图，用于反向传播求导。相当于 JavaScript 的 Promise 链（从后往前算）。

---

## 10. GPU 加速

```python
# 检测 GPU 是否可用
print(torch.cuda.is_available())  # True 或 False

# 指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把模型和数据移到 GPU
model = model.to(device)            # 模型移到 GPU
x = x.to(device)                    # 数据也移到 GPU
output = model(x)                   # 在 GPU 上计算

# 如果有多个 GPU（并行）
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # 自动分配到多个 GPU
```

**类比**：就像把计算从主线程移到 Web Worker，但 GPU 有上千个核心，比 Worker 快得多。

---

## 11. 一个完整的训练循环

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 准备数据
X = torch.randn(100, 10)   # 100个样本，10维特征
y = torch.randn(100, 1)    # 100个标签

# 2. 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

model = Net()
loss_fn = nn.MSELoss()             # 均方误差
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # AdamW 优化器

# 3. 训练循环
for step in range(100):
    pred = model(X)                    # 前向传播
    loss = loss_fn(pred, y)           # 计算损失

    optimizer.zero_grad()              # 清零梯度
    loss.backward()                    # 反向传播（算梯度）
    optimizer.step()                   # 更新参数

    if step % 20 == 0:
        print(f"Step {step}, loss = {loss.item():.4f}")
```

---

## 12. 本项目的训练循环对应关系

```python
# 对应本项目 tinyllm/train.py 的训练循环：

for step in pbar:
    x, y = batcher.get_batch("train")      # 取数据

    optimizer.zero_grad(set_to_none=True)    # 清梯度
    _, loss = model(x, y)                   # 前向传播 + 算损失
    loss.backward()                          # 反向传播
    torch.nn.utils.clip_grad_norm_(...)      # 梯度裁剪
    optimizer.step()                         # 更新参数
```

这和上面"完整训练循环"的区别只是多了：
- `batcher.get_batch()` 取数据
- `_, loss = model(x, y)` 返回 logits 和 loss
- `clip_grad_norm_()` 梯度裁剪防止梯度爆炸
