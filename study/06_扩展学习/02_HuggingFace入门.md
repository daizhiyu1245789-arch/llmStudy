# Hugging Face 入门

> Hugging Face（简称 HF）是全球最大的开源模型库。本篇教你用 HF 的模型和工具。

---

## 1. Hugging Face 是什么？

| | NPM | Hugging Face |
|---|---|---|
| 是什么 | Node.js 包管理器 | AI 模型市场 |
| 托管内容 | JavaScript / Node.js 包 | 预训练模型 |
| 命令行 | `npm install xxx` | `huggingface-cli download xxx` |
| 网站 | npmjs.com | huggingface.co |
| 社区 | 开发者分享包 | 开发者分享模型 |

本项目的 tinyllm 训练的是"自己的小模型"。HF 上的模型是"预训练好的大模型"，可以直接用。

---

## 2. 安装 Hugging Face 工具

```bash
pip install transformers datasets huggingface_hub
```

- `transformers`：使用预训练模型
- `datasets`：加载公开数据集
- `huggingface_hub`：下载/上传模型

---

## 3. 用预训练模型做推理（最基础用法）

### 3.1 GPT-2 生成文本

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 GPT-2（不需要训练，直接用）
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 编码
input_ids = tokenizer.encode("今天天气", return_tensors="pt")

# 生成
output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=0.9,
    top_k=50,
)

# 解码
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)
```

### 3.2 BERT 文本分类

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载微调过的情感分类模型
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# 分类
text = "This movie is great!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
pred = outputs.logits.argmax(dim=-1).item()
print(f"情感: {'正面' if pred >= 3 else '负面'} ({pred}星)")
```

---

## 4. 本项目的模型怎么转成 Hugging Face 格式

如果想把本项目的 GPT 发布到 HF，需要写一个转换脚本：

```python
# convert_to_hf.py
from transformers import GPT2Config, GPT2LMHeadModel
from tinyllm.model import GPT, GPTConfig
from tinyllm.checkpoint import load_checkpoint

# 1. 加载本项目的 checkpoint
ckpt = load_checkpoint("checkpoints/best.pt", map_location="cpu")
tinyllm = GPT(GPTConfig(**ckpt["model_cfg"]))
tinyllm.load_state_dict(ckpt["model_state"])

# 2. 创建对应的 Hugging Face 模型
hf_config = GPT2Config(
    vocab_size=tinyllm.cfg.vocab_size,
    n_positions=tinyllm.cfg.block_size,
    n_ctx=tinyllm.cfg.block_size,
    n_layer=tinyllm.cfg.n_layer,
    n_head=tinyllm.cfg.n_head,
    n_embd=tinyllm.cfg.n_embd,
)
hf_model = GPT2LMHeadModel(hf_config)

# 3. 复制权重（字段名映射）
# 需要把 tinyllm 的字段名映射到 HF 的字段名
state = tinyllm.state_dict()
hf_state = {}

name_map = {
    "wte.weight": "wte.weight",
    "wpe.weight": "wpe.weight",
    "ln_f.weight": "ln_f.weight",
    "ln_f.bias": "ln_f.bias",
    # ... 更多映射
}

for old_name, new_name in name_map.items():
    hf_state[new_name] = state[old_name]

hf_model.load_state_dict(hf_state, strict=False)

# 4. 保存
hf_model.save_pretrained("./hf_tinyllm")
tokenizer.save_pretrained("./hf_tinyllm")
```

---

## 5. 上传模型到 Hugging Face Hub

```bash
# 1. 注册并获取 API Token
# https://huggingface.co/settings/tokens

# 2. 登录
huggingface-cli login
# 输入你的 Token

# 3. 上传模型
python convert_to_hf.py  # 先生成 HF 格式
```

```python
from huggingface_hub import create_repo, upload_folder

# 创建仓库
create_repo("your-username/tinyllm-demo", exist_ok=True)

# 上传
upload_folder(
    repo_id="your-username/tinyllm-demo",
    folder_path="./hf_tinyllm",
    repo_type="model",
)
```

之后别人可以这样用你的模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/tinyllm-demo")
tokenizer = AutoTokenizer.from_pretrained("your-username/tinyllm-demo")
```

---

## 6. 用 Hugging Face 的模型微调自己的任务

HF 的 pipeline 让微调变得很简单：

```python
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# 加载数据集（HF 上有很多公开数据集）
dataset = load_dataset("rotten_tomatoes")  # 电影评论情感分类

# 加载预训练模型
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

# 训练
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

trainer.train()
```

---

## 7. 本项目和 Hugging Face 的关系

| | 本项目 tinyllm | Hugging Face Transformers |
|---|---|---|
| 目的 | 从零学习原理 | 生产级使用 |
| 模型规模 | 几十 MB | 几 MB ~ 几百 GB |
| 预训练 | 需要自己训练 | 直接下载用 |
| 微调 | 自己实现 | 内置 fine-tune 接口 |
| 部署 | 自己包装 API | 内置推理 API |
| 代码 | 全部透明 | 高度封装 |

**建议**：用本项目学原理，用 HF 做生产。

---

## 8. 常用 HF 模型推荐

| 模型 | 用途 | 大小 |
|------|------|------|
| gpt2 | 文本生成 | 500MB |
| distilgpt2 | 轻量文本生成 | 250MB |
| bert-base-uncased | 文本分类/问答 | 400MB |
| t5-small | 翻译/摘要 | 300MB |
| whisper-small | 语音识别 | 300MB |
| stable-diffusion-... | AI 绘画 | 2-5GB |

---

## 9. 你现在做到的事情

```
✅ 学会用 Hugging Face Hub 上的预训练模型
✅ 理解本项目和 HF 的关系
✅ 学会把模型转成 HF 格式
✅ 学会上传模型到 HF
✅ 了解如何微调预训练模型

恭喜！你已经从"学深度学习"进入"用深度学习"的阶段了！
```

---

## 扩展：继续深入的方向

```
1. 更复杂的模型架构
   - LLaMA / Mistral 等开源大模型
   - LoRA / QLoRA 高效微调

2. 多模态
   - 图像生成（Stable Diffusion）
   - 视觉语言（GPT-4V）
   - 语音识别（Whisper）

3. 生产部署
   - vLLM 高效推理
   - TensorRT 优化
   - ONNX 跨平台部署

4. 应用开发
   - LangChain（构建 LLM 应用）
   - RAG（检索增强生成）
   - Agent（AI 代理）
```
