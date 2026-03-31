# tinyllm

从 0 开始实现一个“麻雀虽小、五脏俱全”的极小语言模型（GPT 风格因果 Transformer），具备：

- 字符级分词（默认，保证生成可读；不依赖第三方 tokenizer）
- 字节级分词（可选，适配任意二进制/乱码文本）
- 训练/验证数据切分与随机批采样
- 训练脚本（AdamW、梯度裁剪、断点保存、可恢复训练）
- 推理生成脚本（temperature、top-k）

## 安装

```bash
python -m pip install -r requirements.txt
```

## 快速开始（CPU 小跑）

准备一份文本语料（默认提供 `data/sample.txt`）：

```bash
python -m tinyllm.train --data_path data/sample.txt --max_steps 200 --log_interval 20 --eval_interval 100 --save_interval 100
```

如需切换到字节级分词：

```bash
python -m tinyllm.train --data_path data/sample.txt --tokenizer byte
```

生成：

```bash
python -m tinyllm.generate --ckpt_path checkpoints/latest.pt --prompt "今天" --max_new_tokens 120 --temperature 0.9 --top_k 50
```

## 训练你自己的语料

把任意 `.txt` 文本路径传给 `--data_path` 即可。模型默认是非常小的配置，适合验证端到端流程；想提升效果可增大：

- `--n_layer` / `--n_head` / `--n_embd`
- `--block_size`
- `--max_steps`

## 目录结构

### tinyllm/model.py — GPT 模型核心定义

实现完整的 GPT-2 风格因果 Transformer，包含以下组件：

- **GPTConfig**：模型超参数配置数据类（vocab_size、block_size、n_layer、n_head、n_embd、dropout、bias）
- **CausalSelfAttention**：因果自注意力层。每个头独立计算注意力分数，通过下三角掩码（causal mask）屏蔽未来位置；支持 PyTorch 2.0+ 的融合版本 `F.scaled_dot_product_attention`，也支持手动实现作为降级方案
- **MLP**：前馈神经网络层（FFN）。结构为 `n_embd → 4*n_embd → n_embd`，中间使用 GELU 激活函数
- **Block**：一个完整的 Transformer 编码器块。先做 LayerNorm，再过注意力/MLP，最后用残差连接相加（Pre-Norm 风格）
- **GPT**：主模型。包含 Token Embedding（词表嵌入）、Position Embedding（位置嵌入）、N 个 Block 堆叠、最终 LayerNorm、Language Model Head（LM Head）；LM Head 与 Token Embedding 共享权重（tie_weights）
- **GPT.generate()**：自回归生成方法。逐步预测下一个 token，支持温度采样（temperature）和 Top-K 过滤

### tinyllm/data.py — 数据管道与批次采样

负责将原始文本语料转换为模型可用的训练批次：

- **DataConfig**：数据配置数据类，包含 block_size（上下文窗口长度）、batch_size（批大小）、train_split（训练/验证集比例，默认 0.9）
- **TextBatcher**：文本批次采样器。将整个语料（1D token ID 数组）切分为训练集和验证集；`get_batch(split)` 方法从指定集合中随机采样一个批次，返回 `(x, y)` 元组，其中 y 是 x 偏移 1 位的结果（自回归语言模型的标准做法）

### tinyllm/tokenizer.py — 分词器

提供两种分词策略，均不依赖第三方库：

- **TokenizerProtocol**：分词器的抽象接口，定义 `encode()` 和 `decode()` 方法
- **ByteTokenizer**：字节级分词器。将 UTF-8 编码后的每个字节（0-255）作为一个 token，词表固定为 256；可处理任意文本和二进制数据
- **CharTokenizer**：字符级分词器。将每个唯一字符作为 token，词表由训练语料决定；支持 `train()` 从文本训练、`encode()` 编码、`decode()` 解码
- **tokenizer_to_extra / tokenizer_from_extra**：分词器的序列化/反序列化函数，用于将分词器配置存入 checkpoint 并恢复

### tinyllm/train.py — 训练入口

完整的训练脚本，实现了端到端训练流程：

- `_get_device()`：根据 "auto" / "cpu" / "cuda" 等字符串解析设备
- `estimate_loss()`：周期性评估模型在训练集和验证集上的损失（关闭 Dropout，多次采样取平均）
- **main()**：主训练函数。命令行参数覆盖所有超参数（模型架构、训练步数、学习率、梯度裁剪阈值等）；支持断点恢复（`--resume`）、混合精度训练（`--dtype bf16`）、梯度裁剪（AdamW 优化器）；每逢 `eval_interval` 评估一次，`save_interval` 保存增量 checkpoint，始终保持 `latest.pt` 和最优 `best.pt`

### tinyllm/generate.py — 推理生成入口

基于训练好的模型进行文本生成：

- `_get_device()`：设备解析（同 train.py）
- **main()**：从指定 checkpoint 加载模型和分词器，将 `--prompt` 编码为 token ID，调用 `model.generate()` 自回归生成新 token，最后将生成结果解码为文本并打印；支持 `--temperature` 控制随机性、`--top_k` 过滤高概率 token

### tinyllm/checkpoint.py — 断点保存与加载

管理训练过程中的模型存档：

- **save_checkpoint()**：将模型权重（state_dict）、优化器状态、GPTConfig 超参数、当前步数、最佳验证损失、分词器配置等写入 .pt 文件
- **load_checkpoint()**：从磁盘加载 checkpoint 并将张量映射到目标设备（CPU/CUDA）
- **build_model_from_checkpoint()**：从 checkpoint 恢复完整的 GPT 模型（配置 + 权重），用于推理或训练恢复


