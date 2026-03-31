# Tokenizer 分词器详解

> 在深度学习模型处理文本之前，必须先把文字变成数字。这个转换过程就叫做"分词"（Tokenization）。

---

## 1. 什么是 Token？—— 就像 URL 路径里的 path segment

Node.js 的 URL 解析：

```
https://api.example.com/v1/users/123/profile
                            ↓
Segments: ["v1", "users", "123", "profile"]
                            ↓
每个 segment 都是一个"语义单元"
```

Token 就是文本的**最小语义单元**：

```
"今天天气很好" → ["今", "天", "天", "气", "很", "好"]   （按字符切分）
                            ↓
Token IDs:            [5, 23, 23, 42, 88, 102]
```

模型只能处理数字，不能直接处理文字。所以要先把文字变成数字 ID。

---

## 2. 三种常见的 Token 切分策略

### 策略一：字符级（Character-level）

把每个字符当作一个 token：

```
"你好世界" → ['你', '好', '世', '界'] → [3, 15, 42, 88]
```

| 优点 | 缺点 |
|------|------|
| 词表小（通常几百～几千个字符） | 序列很长（中英文都是一个字一个 token） |
| 不需要训练分词器 | 相邻字符之间没有天然边界 |
| 能处理任意文本 |  |

### 策略二：词级（Word-level）

按词语切分：

```
"今天天气很好" → ['今天', '天气', '很好'] → [1024, 2048, 889]
```

| 优点 | 缺点 |
|------|------|
| 每个 token 语义完整 | 词表巨大（英文可达百万） |
| 序列短 | OOV（未登录词）问题严重 |
| 语义清晰 | 需要预处理器（spaCy、jieba 等） |

### 策略三：子词级（Subword-level，BPE / WordPiece）

把词切成更小的片段：

```
"今天天气很好"
  ↓ 假设"天气"是常用词，不切
  ↓ "很"也是常用词，不切
["今", "天", "天气", "很", "好"] → [5, 12, 2048, 88, 102]
```

常见算法：
- **BPE**（Byte Pair Encoding）：统计高频字符对，合并它们
- **WordPiece**：基于语言模型选择合并

GPT-2 / BERT 用的是 BPE，OpenAI 的 tiktoken 是 BPE 的高效实现。

### 本项目：两种都支持

```python
# CharTokenizer：字符级
"今天" → [5, 12]  # 每个字符一个 ID

# ByteTokenizer：字节级
"今天" → [228, 189, 160, 233, 156, 136]  # UTF-8 字节序列
```

---

## 3. Tokenizer 的三个核心操作

### 3.1 构建词表（train）

```js
// CharTokenizer.train() 的逻辑：
function train(text) {
  const chars = unique(sorted(text)); // 统计出现过的所有不同字符
  const vocab = ['<unk>', ...chars];   // <unk> 是未知字符的兜底
  return vocab; // vocab[0] = '<unk>', vocab[1] = 第一个字符
}
```

### 3.2 编码（encode）

```js
// 把文字变成数字 ID
function encode(text, vocab) {
  const stoi = {};
  vocab.forEach((char, index) => stoi[char] = index);

  return text.split('').map(char => stoi[char] ?? stoi['<unk>']);
}

encode("你好", ['<unk>', '你', '好']);
// → [1, 2]
```

### 3.3 解码（decode）

```js
// 把数字 ID 变回文字
function decode(ids, vocab) {
  return ids.map(id => vocab[id] ?? '<unk>').join('');
}

decode([1, 2], ['<unk>', '你', '好']);
// → "你好"
```

---

## 4. 本项目两种 Tokenizer 对比

```python
# ByteTokenizer
text = "Hello 你好"  # 英文 + 中文
bytes_data = text.encode("utf-8")  # → b'Hello \xe4\xb8\xad\xe6\x96\x87'
token_ids = list(bytes_data)      # → [72, 101, 108, 108, 111, 32, 228, 189, 160, 233, 156, 136]
# 词表固定为 256（每个字节 0-255）
```

| | CharTokenizer | ByteTokenizer |
|---|---|---|
| 词表大小 | 取决于语料（几百～几千） | 固定 256 |
| 编码方式 | 字符 → ID | UTF-8 字节 → ID |
| 中文序列长度 | 短（每个汉字 1 个 token） | 长（每个汉字 2-4 个 token） |
| 能否处理乱码 | 不能（遇到未知字符返回 `<unk>`） | 能（字节级总能找到对应 ID） |
| 是否需要训练 | 需要（扫描语料统计字符） | 不需要（词表固定） |

---

## 5. Tokenizer 和模型的关系

```
文字 → Tokenizer.encode() → Token IDs → 模型.forward() → Output IDs → Tokenizer.decode() → 文字
```

本项目的 GPT 模型：
- `model.wte`（Token Embedding）：Token ID → n_embd 维向量（查表）
- `model.lm_head`（LM Head）：n_embd 维向量 → vocab_size 维 logit（预测每个词的概率）

Embedding 和 LM Head 共享权重（`model.lm_head.weight = model.wte.weight`）：
```
logits = x @ lm_head.weight.T
       = x @ wte.weight.T
       → Embedding 的转置
```
这是一种常见技巧，节省了一半的 Embedding 层参数量。

---

## 6. 词表大小对模型的影响

```
词表太小（如 32）→ 大量未知字符 → 模型只能处理极少数情况
词表太大（如 100000）→ 每个 token 的统计数据稀疏 → 泛化差

词表大小需要平衡：
- 足够大：覆盖绝大多数常见词/字
- 足够小：每个 token 有足够的训练样本
```

本项目默认：
- 字符级（CharTokenizer）：词表 = 训练语料中出现的字符种类数 + 1（unk）
- 字节级（ByteTokenizer）：词表 = 256（固定）
