# Tokenizer 分词器详解

> 文本进入模型之前，必须先变成数字。这个转换过程叫"分词"（Tokenization）。

---

## 1. 为什么需要 Tokenizer？

模型只认识数字，不认识文字。

```
文字 → Tokenizer → 数字 ID → 模型 → 数字 ID → Tokenizer → 文字
```

就像：
```
HTTP 请求 → 路由解析 → handler → 响应
```

---

## 2. 文本 → 数字的三个步骤

### 步骤 1：切分（Tokenize）

把文本切成小片段。

```
"今天天气很好"
  ↓ 字符级切分
["今", "天", "天", "气", "很", "好"]
```

### 步骤 2：查表（Encode）

每个片段 → 对应一个整数 ID。

```python
vocab = ["<unk>", "今", "天", "气", "很", "好", ...]
# 索引:      0      1    2    3    4    5

"今天天气很好"
  ↓
[1, 2, 2, 3, 4, 5]
```

### 步骤 3：查表（Decode）

数字 ID → 还原回文本。

```python
[1, 2, 2, 3, 4, 5]
  ↓
["今", "天", "天", "气", "很", "好"]
  ↓
"今天天气很好"
```

---

## 3. 三种切分策略

### 策略 1：字符级（Character-level）

每个字符 = 一个 token。

```
"你好世界" → ['你', '好', '世', '界'] → [3, 15, 42, 88]
```

| 优点 | 缺点 |
|------|------|
| 词表小 | 序列很长（中文每个字一个） |
| 不需要预分词器 | 相邻字之间没有天然边界 |
| 能处理任意文本 | |

### 策略 2：词级（Word-level）

每个词 = 一个 token。

```
"今天天气很好" → ['今天', '天气', '很好'] → [1024, 2048, 889]
```

| 优点 | 缺点 |
|------|------|
| 语义完整 | 词表巨大（英文可达百万） |
| 序列短 | OOV（未登录词）问题 |
| 语义清晰 | 需要分词器（jieba 等） |

### 策略 3：子词级（Subword-level）

把词切成更小的片段。

```
"今天天气很好"
  ↓
["今", "天", "天气", "很", "好"] → [5, 12, 2048, 88, 102]
```

常见算法：BPE（Byte Pair Encoding）、WordPiece。
GPT-2 用的是 BPE。

---

## 4. 本项目的两种 Tokenizer

### ByteTokenizer（字节级）

每个 UTF-8 字节 = 一个 token，词表固定为 256。

```python
# "中" 的 UTF-8 编码是 3 个字节
text = "中"
bytes_data = text.encode("utf-8")  # b'\xe4\xb8\xad'
token_ids = list(bytes_data)       # [228, 189, 160]

# 可以处理任意文本
# 能处理乱码（二进制数据也没问题）
```

| 特点 | 说明 |
|------|------|
| 词表大小 | 固定 256 |
| 序列长度 | 中文 2-4 倍（因为 UTF-8 变长） |
| 优点 | 可处理任意输入，不会 OOV |

### CharTokenizer（字符级）

每个不同字符 = 一个 token，词表由语料决定。

```python
# 从语料训练得到词表
tokenizer = CharTokenizer.train(text)

# 词表 = 所有出现过的不同字符
# itos[0] = "<unk>"（未知字符兜底）
# itos[1:] = 所有字符

# 编码
ids = tokenizer.encode("今天天气很好")  # [3, 4, 4, 5, 6, 7]

# 解码
text = tokenizer.decode(ids)  # "今天天气很好"
```

---

## 5. Embedding 和 LM Head 的关系

```
Token ID → Embedding 表 → n_embd 维向量
                                    ↓
                              Transformer 处理
                                    ↓
                              n_embd 维向量 → LM Head → vocab_size 维 logit
```

本项目的技巧：**Embedding 和 LM Head 共享权重**。

```python
# model.py 的实现
self.wte = nn.Embedding(vocab_size, n_embd)   # Token Embedding
self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

# 共享权重
self.lm_head.weight = self.wte.weight  # LM Head 的权重 = Embedding 的转置
```

好处：节省一半的 Embedding 参数量。
坏处：Input 和 Output 的嵌入表示必须一样（有些情况下不利于学习）。

---

## 6. 词表大小对生成的影响

```
词表太小 → 大量字符归为 <unk> → 生成质量差
词表太大 → 每个 token 的统计数据稀疏 → 泛化差

词表大小需要平衡覆盖度和统计充分性。
```

本项目配置：
- 字符级：词表 ≈ 训练语料的不同字符数（通常几百到几千）
- 字节级：词表固定 256

---

## 7. Tokenizer 和模型的关系图

```
                    原始文本
                       ↓
              CharTokenizer.encode()
              ByteTokenizer.encode()
                       ↓
                 Token IDs
                [5, 23, 42, ...]
                       ↓
              model.wte (Embedding)
                 Token IDs → 向量
                       ↓
                 Transformer Block × N
                       ↓
               model.ln_f (LayerNorm)
                       ↓
               model.lm_head (输出投影)
                       ↓
               Logits（每个 token 的词表概率）
              [0.01, 0.03, 0.8, ...] （Softmax 前）
                       ↓
                采样 / 贪婪选择
                       ↓
               Next Token ID
                       ↓
              CharTokenizer.decode()
              ByteTokenizer.decode()
                       ↓
                    生成文本
```
