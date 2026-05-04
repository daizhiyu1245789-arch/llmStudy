# Python 和 PyTorch 基础练习

> 对应 `study/02_Python和PyTorch基础/`。目标不是成为 Python 专家，而是看懂 tinyllm 代码、能跑训练脚本、能改一点模型参数。

---

## 01_Python vs JS 语法对比

对应章节：

```text
02_Python和PyTorch基础/01_Python_vs_JS语法对比.md
```

### 基础练习

- [ ] 写出 Python 里的 `list`、`dict`、`tuple`
- [ ] 写出 Python 的 `if / elif / else`
- [ ] 写出 `for item in items`
- [ ] 写出列表推导式
- [ ] 解释 Python 缩进为什么重要

### 实战练习

把下面 JS 思路改成 Python：

```js
const users = [
  { name: "a", score: 80 },
  { name: "b", score: 95 },
];
const passed = users.filter((u) => u.score >= 90).map((u) => u.name);
```

要求：

- [ ] 用 Python list/dict 表示数据
- [ ] 用列表推导式筛选
- [ ] 打印结果

---

## 02_Python 独有概念

对应章节：

```text
02_Python和PyTorch基础/02_Python独有概念.md
```

### 基础练习

- [ ] 解释 `with open(...)` 的作用
- [ ] 解释装饰器大概是什么
- [ ] 解释 `__name__ == "__main__"`
- [ ] 解释模块导入

### 实战练习

写一个 Python 文件：

```text
read_sample.py
```

要求：

- [ ] 用 `with open` 读取 `data/sample.txt`
- [ ] 打印前 100 个字符
- [ ] 写 `main()` 函数
- [ ] 用 `if __name__ == "__main__"` 调用

---

## 03_PyTorch 快速入门

对应章节：

```text
02_Python和PyTorch基础/03_PyTorch快速入门.md
```

### 基础练习

- [ ] 创建一个 `[3, 4]` 的随机 tensor
- [ ] 打印 shape
- [ ] 做矩阵乘法
- [ ] 使用 `nn.Linear`
- [ ] 写一个最小 `nn.Module`

### 实战练习

写一个简单模型：

```python
class TinyClassifier(nn.Module):
    ...
```

要求：

- [ ] 输入维度 10
- [ ] 隐藏层 32
- [ ] 输出维度 2
- [ ] forward 里用 ReLU
- [ ] 跑一次假输入 `[4, 10]`
- [ ] 输出 shape 是 `[4, 2]`

---

## 04_动手跑第一个模型

对应章节：

```text
02_Python和PyTorch基础/04_动手跑第一个模型.md
```

### 基础练习

- [ ] 能运行 tinyllm 的训练命令
- [ ] 知道 `--data_path` 是什么
- [ ] 知道 `--max_steps` 是什么
- [ ] 知道 checkpoint 保存在哪里

### 实战练习

完成一次最小训练：

```bash
python -m tinyllm.train --data_path data/sample.txt --max_steps 50 --log_interval 10 --eval_interval 20 --save_interval 20
```

然后：

- [ ] 截取一次 loss 日志
- [ ] 运行 generate
- [ ] 修改 prompt 再生成一次
- [ ] 写下两次输出差异

### 验收标准

你能解释：

```text
训练是更新模型参数，生成是使用模型参数。
```
