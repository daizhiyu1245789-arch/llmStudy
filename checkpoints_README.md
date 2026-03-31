# Checkpoints 说明文档

`checkpoints/` 目录下保存的是训练过程中生成的模型"存档"文件，记录了模型在各个训练阶段的完整状态。

---

## 文件列表

| 文件 | 含义 | 更新时机 |
|------|------|---------|
| `latest.pt` | **最新存档** | 每次评估（eval_interval）时覆盖更新，始终反映最近一次训练状态 |
| `best.pt` | **最优存档** | 每当当前验证损失低于历史最佳时覆盖，确保保存的是表现最好的模型 |
| `step_15.pt` | 第 15 步存档 | 按 save_interval 周期性保存 |
| `step_30.pt` | 第 30 步存档 | 同上 |

---

## checkpoint 文件内部结构

每个 `.pt` 文件本质上是一个 Python 字典（通过 `torch.save` 序列化存储），包含以下字段：

```python
{
    "step": int,                  # 当前是第几步（从 1 开始计数）
    "best_val_loss": float,      # 历史上最佳的验证集损失（用于判断是否更新 best.pt）
    "model_cfg": {                # GPTConfig 超参数的字典形式
        "vocab_size": 256,        # 词表大小
        "block_size": 128,        # 上下文窗口长度
        "n_layer": 4,             # Transformer 层数
        "n_head": 4,              # 注意力头数
        "n_embd": 128,            # 嵌入向量维度
        "dropout": 0.1,           # Dropout 概率
        "bias": True,             # 是否使用偏置
    },
    "model_state": dict,          # 模型所有权重（state_dict）
                                 # 包含：Embedding 权重、Linear 层权重和偏置、
                                 # LayerNorm 的 scale 和 bias、buffers 等
    "optimizer_state": dict,      # AdamW 优化器的内部状态
                                 # 包含：一阶动量（exp_avg）、
                                 # 二阶动量（exp_avg_sq）、学习率等
    "extra": {                    # 额外信息，通常存放分词器配置
        "tokenizer": {
            "type": "char",       # 分词器类型："char" 或 "byte"
            "itos": [...],        # CharTokenizer 特有：索引到字符的映射表
            "unk_token": "<unk>"  # CharTokenizer 特有：未知字符的替代符号
        }
    }
}
```

---

## 各字段详解

### step — 训练步数

当前存档对应的是训练过程中的第几步。这个数字用于：
- 恢复训练时知道从哪里继续
- 判断是否到达 max_steps 终止训练
- 为中间存档文件命名（如 step_15.pt）

### best_val_loss — 最佳验证损失

模型在验证集上的损失值，越低表示泛化能力越强。每次保存 `best.pt` 时会用当前验证损失和这个值比较，如果更小则覆盖。用于：
- 最终推理时选择使用哪个模型
- 了解模型能力的上限

### model_cfg — 模型配置

GPTConfig 的字典形式，记录了模型的所有超参数。从 checkpoint 加载模型时会用这些参数重建模型结构，确保配置一致。

### model_state — 模型权重

`model.state_dict()` 的返回值，是一个有序字典。每个键是层的名字（如 `wte.weight`、`blocks.0.attn.qkv.weight`），每个值是对应的 PyTorch 张量。包含：

| 层类型 | 保存的权重 |
|--------|-----------|
| nn.Embedding | token embedding 矩阵（vocab_size × n_embd）、position embedding 矩阵（block_size × n_embd） |
| nn.Linear | weight（输出维度 × 输入维度）、bias（输出维度） |
| nn.LayerNorm | scale（归一化参数）、bias（偏置，如果 bias=True） |
| buffers | **不包含**，causal_mask 等 buffer 以 `persistent=False` 注册，不进入 state_dict |

### optimizer_state — 优化器状态

`optimizer.state_dict()` 的返回值，包含 AdamW 算法的内部状态变量：

- **exp_avg**：梯度的一阶指数移动平均（类似动量），用于平滑更新方向
- **exp_avg_sq**：梯度的二阶指数移动平均（用于自适应学习率）
- **step**：当前步数计数器

恢复训练时加载 optimizer_state，可以确保优化器从中断前的状态继续，而不是从头开始——这对于使用 AdamW 自适应学习率的训练至关重要。

### extra — 分词器配置

训练时将分词器配置序列化存储，恢复训练或推理时用来重建分词器：

- **type = "char"**：字符级分词器，需要存储 `itos`（索引→字符映射表）和 `unk_token`
- **type = "byte"**：字节级分词器，词表固定为 256，不需要额外存储

---

## 使用场景

### 1. 恢复中断的训练

```bash
python -m tinyllm.train \
    --data_path data/sample.txt \
    --resume
```

加上 `--resume` 参数后，脚本会自动读取 `checkpoints/latest.pt`，恢复：
- 模型权重
- 优化器状态（从中断处继续，AdamW 的动量等信息保持连贯）
- 分词器配置
- 当前 step 数（从断点继续计数）

### 2. 用最佳模型生成文本

```bash
python -m tinyllm.generate \
    --ckpt_path checkpoints/best.pt \
    --prompt "今天" \
    --max_new_tokens 120 \
    --temperature 0.9 \
    --top_k 50
```

### 3. 观察模型在不同训练阶段的变化

```bash
# 用第 15 步的模型生成
python -m tinyllm.generate --ckpt_path checkpoints/step_15.pt --prompt "今天"

# 用第 30 步的模型生成
python -m tinyllm.generate --ckpt_path checkpoints/step_30.pt --prompt "今天"
```

对比不同 step 的输出，可以直观观察到模型从"胡说八道"到"逐渐学会语言"的变化过程。

### 4. 迁移学习或微调

将 `best.pt` 或 `latest.pt` 作为预训练权重加载，在此基础上针对新语料微调。

---

## 文件命名规则

- `best.pt`：始终存放验证损失最低的模型
- `latest.pt`：始终存放最近一次评估时的模型状态
- `step_N.pt`：每逢 save_interval 步保存一次，用于保留中间过程

训练结束后，`step_N.pt` 文件可以删除以节省空间，只保留 `best.pt` 和 `latest.pt` 即可。
