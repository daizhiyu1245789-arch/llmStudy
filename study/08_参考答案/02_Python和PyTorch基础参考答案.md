# Python 和 PyTorch 基础参考答案

> 对应 `07_练习/02_Python和PyTorch基础练习.md`。

---

## 01_Python vs JS 语法对比

### JS filter + map 改 Python

JS：

```js
const users = [
  { name: "a", score: 80 },
  { name: "b", score: 95 },
];
const passed = users.filter((u) => u.score >= 90).map((u) => u.name);
```

Python：

```python
users = [
    {"name": "a", "score": 80},
    {"name": "b", "score": 95},
]

passed = [user["name"] for user in users if user["score"] >= 90]

print(passed)  # ["b"]
```

关键点：

```text
JS 对象 → Python dict
JS 数组 → Python list
filter + map → Python 列表推导式
```

---

## 02_Python 独有概念

### read_sample.py

```python
from pathlib import Path


def main() -> None:
    path = Path("data/sample.txt")

    with path.open("r", encoding="utf-8") as f:
        text = f.read()

    print(text[:100])


if __name__ == "__main__":
    main()
```

解释：

- `Path`：更现代的路径处理
- `with`：自动关闭文件
- `encoding="utf-8"`：避免中文乱码
- `main()`：让脚本结构清楚
- `if __name__ == "__main__"`：只有直接运行该文件时才执行

---

## 03_PyTorch 快速入门

### Tensor 基础

```python
import torch

x = torch.randn(3, 4)
print(x.shape)  # torch.Size([3, 4])

w = torch.randn(4, 5)
y = x @ w
print(y.shape)  # torch.Size([3, 5])
```

### TinyClassifier

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


model = TinyClassifier()
fake_input = torch.randn(4, 10)
output = model(fake_input)

print(output.shape)  # torch.Size([4, 2])
```

验收：

```text
输入 [4, 10]
输出 [4, 2]
```

---

## 04_动手跑第一个模型

### 训练命令

```bash
python -m tinyllm.train \
  --data_path data/sample.txt \
  --max_steps 50 \
  --log_interval 10 \
  --eval_interval 20 \
  --save_interval 20
```

你应该看到：

```text
step ... loss ...
```

loss 不一定每一步都下降，但整体应该有下降趋势。

### 生成命令

```bash
python -m tinyllm.generate \
  --ckpt_path checkpoints/latest.pt \
  --prompt "今天" \
  --max_new_tokens 80 \
  --temperature 0.8
```

### temperature 对比答案参考

```text
temperature=0.3：
更保守，更重复，更稳定。

temperature=1.2：
更随机，可能更有变化，也更容易乱码或跑偏。
```

### 关键解释

```text
训练：更新模型参数
生成：使用已有参数预测下一个 token
checkpoint：保存训练后的模型参数
loss：模型预测错多少
```
