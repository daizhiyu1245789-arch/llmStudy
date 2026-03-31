# 把 tinyllm 项目跑起来

> 这一篇手把手带你在 Windows 上把项目跑起来，看到模型训练和生成。

---

## 环境要求

- Python 3.8+
- 不需要 GPU（CPU 就能跑）
- 约 2GB 空闲磁盘空间（PyTorch 比较大）

---

## 步骤一：安装 Python

如果没有安装 Python，去 https://python.org 下载安装。

安装时勾选：
```
☑ Add Python to PATH（把 Python 加入环境变量）
☑ Install pip（包管理器）
```

验证安装：
```bash
python --version
# 应该看到：Python 3.10.x 或更高版本
```

---

## 步骤二：创建虚拟环境

```bash
# 进入项目目录
cd e:/python/llmStudy

# 创建虚拟环境（隔离依赖）
python -m venv .venv

# Windows 激活虚拟环境
.venv\Scripts\activate

# Linux/Mac：
# source .venv/bin/activate
```

激活成功后，命令行前面会有个 `(.venv)` 标记。

---

## 步骤三：安装依赖

```bash
pip install torch numpy tqdm
```

安装顺序建议：
```
1. 先装 torch（最大，可能需要几分钟）
2. 再装其他小包
```

如果 PyTorch 下载慢，用国内镜像：
```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 步骤四：检查项目文件

```bash
# 查看项目结构
ls tinyllm/
# 应该看到：
# model.py
# data.py
# tokenizer.py
# train.py
# generate.py
# checkpoint.py

ls data/
# 应该看到：
# sample.txt（训练语料）
```

---

## 步骤五：第一次训练

```bash
python -m tinyllm.train \
    --data_path data/sample.txt \
    --max_steps 50 \
    --log_interval 10 \
    --eval_interval 20 \
    --save_interval 20
```

参数说明：
```
--data_path      训练语料的文件路径
--max_steps     训练多少步（50步约1-2分钟CPU）
--log_interval  每多少步打印一次日志
--eval_interval 每多少步评估一次验证集
--save_interval 每多少步保存一次checkpoint
```

---

## 步骤六：理解训练输出

预期输出大概长这样：

```
100%|██████████| 50/50 [01:23<00:00,  1.67s/step]
step 11 loss 3.2812 (3.2s)
step 21 loss 2.9451 (6.1s)
step 31 loss 2.5123 (9.4s)
step 41 loss 2.2034 (12.8s)
```

**loss 是什么？**
- loss = 模型预测的错误程度
- 3.3 → 2.2 → 1.8 ... 逐渐下降 = 模型在学到东西

**loss 降到多少算好？**
- 3.0+：模型基本没学到东西
- 2.0-2.5：初步学会了一些模式
- 1.5-2.0：生成质量还可以
- 1.0 以下：比较流畅

**50 步能降到多少？**
通常只能降到 2.0 左右。想更低需要 200-500 步。

---

## 步骤七：检查 checkpoint

训练完成后，查看 checkpoints 目录：

```bash
ls checkpoints/
```

应该看到：
```
best.pt       ← 最佳模型（验证损失最低）
latest.pt     ← 最新存档
step_20.pt    ← 第20步
step_40.pt    ← 第40步
```

---

## 步骤八：用模型生成文本

```bash
python -m tinyllm.generate \
    --ckpt_path checkpoints/latest.pt \
    --prompt "今天" \
    --max_new_tokens 30 \
    --temperature 0.9 \
    --top_k 50
```

参数说明：
```
--ckpt_path      用哪个checkpoint
--prompt         生成的开头（给模型一个提示）
--max_new_tokens 生成多少个新token
--temperature    随机性（越大越有创意，越小越保守）
--top_k          只从概率最高的k个token里选
```

---

## 步骤九：观察不同参数的输出差异

### 调整 temperature

```bash
# 高随机性（更有创意但可能乱来）
python -m tinyllm.generate --ckpt_path checkpoints/latest.pt --prompt "今天" --temperature 1.5

# 低随机性（更稳定）
python -m tinyllm.generate --ckpt_path checkpoints/latest.pt --prompt "今天" --temperature 0.5

# 贪婪（最保守，直接选概率最高的）
python -m tinyllm.generate --ckpt_path checkpoints/latest.pt --prompt "今天" --temperature 0
```

### 调整 top_k

```bash
# top_k=1 等价于完全贪婪
python -m tinyllm.generate --ckpt_path checkpoints/latest.pt --prompt "今天" --top_k 1

# top_k=50（默认）
python -m tinyllm.generate --ckpt_path checkpoints/latest.pt --prompt "今天" --top_k 50

# top_k=0 或不指定 = 不过滤
python -m tinyllm.generate --ckpt_path checkpoints/latest.pt --prompt "今天" --top_k 0
```

---

## 步骤十：更长的训练

想看更好的效果，跑 200 步：

```bash
python -m tinyllm.train \
    --data_path data/sample.txt \
    --max_steps 200 \
    --log_interval 20 \
    --eval_interval 50 \
    --save_interval 50
```

大约需要 5-10 分钟（CPU）。

然后用最新的 checkpoint 生成：
```bash
python -m tinyllm.generate --ckpt_path checkpoints/latest.pt --prompt "今天" --max_new_tokens 50
```

---

## 常见问题

### Q：loss 一直是 3.3 不下降
A：50 步太少，模型需要更多时间学习。至少跑 200 步观察。

### Q：生成的是乱码
A：正常，loss 还在 2.5 以上时模型还没学明白。继续训练或检查数据是否正常。

### Q：报错 "ModuleNotFoundError"
A：确保激活了虚拟环境（命令行前有 .venv 标记）。

### Q：内存不足
A：减小 batch_size：
```bash
python -m tinyllm.train --data_path data/sample.txt --batch_size 8
```

---

## 训练完成！接下来

```
✅ 安装并运行了 PyTorch
✅ 跑通了完整的训练流程
✅ 用训练好的模型生成了文本
✅ 学会了调参观察效果

下一步 → 03_项目实战/02_用自己的数据训练.md
```
