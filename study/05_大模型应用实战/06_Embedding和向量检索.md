# Embedding 和向量检索

> RAG 的核心不是“把所有文档塞进 prompt”，而是先用 Embedding 找到最相关的文档片段，再把这些片段交给模型回答。

---

## 0. 学完这一篇你会什么？

你会学会：

- Embedding 是什么
- 为什么文本可以变成向量
- 相似度检索是怎么工作的
- 为什么 RAG 需要切块
- Node 项目里怎么组织文档切块
- 向量库和普通数据库的区别
- 做 RAG 前要记录哪些字段

这一篇先讲原理和工程结构，下一篇再把它串成完整 RAG。

---

## 1. Embedding 是什么？

Embedding 可以理解为：

```text
把一段文本变成一组数字向量
```

例如：

```text
"Node.js 怎么接 Hugging Face?"
  ↓ embedding model
[0.12, -0.33, 0.87, ...]
```

这组数字不是随机的，它表示这段文本的语义位置。

语义相近的文本，向量距离更近：

```text
"Node 如何调用大模型 API"
"Express 怎么接 Hugging Face"
```

语义不相关的文本，向量距离更远：

```text
"Node 如何调用大模型 API"
"今天晚饭吃什么"
```

---

## 2. Embedding 和 Tokenizer 的区别

| 概念 | 作用 | 输出 |
|---|---|---|
| Tokenizer | 把文本切成 token ID | `[123, 456, 789]` |
| Embedding | 把文本表示成语义向量 | `[0.12, -0.33, ...]` |

Tokenizer 是模型读文本的入口。

Embedding 是用来做语义比较、检索、聚类的表示。

---

## 3. 为什么需要向量检索？

假设你有 1000 篇文档，用户问：

```text
怎么用 Node 调 Hugging Face 做流式输出？
```

你不能把 1000 篇文档全塞给模型：

- 上下文窗口不够
- 成本太高
- 噪声太多
- 回答容易混乱

正确做法：

```text
用户问题
  ↓
生成 query embedding
  ↓
去向量库找最相似的文档片段
  ↓
取 top 3 / top 5
  ↓
把这些片段塞进 prompt
  ↓
让模型基于资料回答
```

这就是 RAG 的检索部分。

---

## 4. 相似度怎么计算？

常见方法：

```text
cosine similarity：余弦相似度
dot product：点积
euclidean distance：欧氏距离
```

入门先理解 cosine similarity：

```text
两个向量方向越接近，相似度越高。
```

直觉：

```text
同一个主题的文本 → 向量方向接近
不同主题的文本 → 向量方向远
```

伪代码：

```ts
function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

---

## 5. 为什么要文档切块？

你不能直接把整篇文档做 embedding。

原因：

- 文档太长，embedding 模型可能截断
- 一篇文档里有多个主题
- 用户问题通常只对应某几个段落
- 检索粒度太粗会带入大量无关内容

所以要切块：

```text
长文档
  ↓
chunk 1
chunk 2
chunk 3
...
```

每个 chunk 单独生成 embedding。

---

## 6. 怎么切块？

最简单策略：按字符长度切。

```ts
export function chunkText(text: string, chunkSize = 800, overlap = 120): string[] {
  const chunks: string[] = [];
  let start = 0;

  while (start < text.length) {
    const end = Math.min(start + chunkSize, text.length);
    const chunk = text.slice(start, end).trim();

    if (chunk) {
      chunks.push(chunk);
    }

    start += chunkSize - overlap;
  }

  return chunks;
}
```

为什么要 overlap？

假设答案刚好跨越两个 chunk 的边界：

```text
chunk 1: ... Node 集成 Hugging
chunk 2: Face 时需要设置 HF_TOKEN ...
```

如果没有重叠，检索可能丢失上下文。

overlap 可以让相邻 chunk 保留一点共同内容：

```text
chunk 1: ... Node 集成 Hugging Face 时需要设置
chunk 2: Hugging Face 时需要设置 HF_TOKEN ...
```

---

## 7. 更好的切块策略

真实项目里，建议优先按结构切：

| 文档类型 | 切块方式 |
|---|---|
| Markdown | 按标题、段落切 |
| API 文档 | 按接口或小节切 |
| FAQ | 一问一答一块 |
| 代码 | 按函数、类、文件切 |
| PDF | 先抽文本，再按段落切 |

切块原则：

```text
一个 chunk 最好表达一个相对完整的意思。
```

不要太短：

```text
上下文不足，模型看不懂。
```

不要太长：

```text
检索不准，占 prompt。
```

入门建议：

```text
chunkSize: 500-1000 中文字符
overlap: 80-150 中文字符
topK: 3-5
```

---

## 8. 向量库里存什么？

不要只存向量。

建议每个 chunk 存：

```ts
type DocumentChunk = {
  id: string;
  documentId: string;
  title: string;
  content: string;
  embedding: number[];
  source: string;
  chunkIndex: number;
  createdAt: string;
};
```

字段说明：

| 字段 | 作用 |
|---|---|
| id | chunk 唯一 ID |
| documentId | 属于哪篇文档 |
| title | 文档标题 |
| content | chunk 原文 |
| embedding | 语义向量 |
| source | 来源路径/URL |
| chunkIndex | 第几个 chunk |
| createdAt | 创建时间 |

为什么要存 source？

因为 RAG 回答最好带引用来源，否则用户不知道答案从哪来。

---

## 9. 向量库有哪些？

| 类型 | 工具 | 适合 |
|---|---|---|
| 本地内存 | 数组 + cosine | 学习、demo |
| SQLite 扩展 | sqlite-vec | 小型本地应用 |
| Postgres | pgvector | 业务系统常用 |
| 专用向量库 | Qdrant / Milvus / Weaviate | 大规模检索 |
| 托管服务 | Pinecone 等 | 快速上线 |

学习阶段建议：

```text
先用内存数组把链路跑通。
再换 pgvector / Qdrant。
```

别一开始就陷入向量数据库选型。

---

## 10. Node 里的最小内存向量库

```ts
type VectorRecord = {
  id: string;
  content: string;
  embedding: number[];
  source?: string;
};

const records: VectorRecord[] = [];

export function addRecord(record: VectorRecord) {
  records.push(record);
}

export function search(queryEmbedding: number[], topK = 5) {
  return records
    .map((record) => ({
      ...record,
      score: cosineSimilarity(queryEmbedding, record.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
```

这个不能直接用于生产，但非常适合学习 RAG 的主流程。

---

## 11. Embedding 模型怎么选？

选择标准：

- 是否支持中文
- 向量维度多少
- 成本
- 延迟
- 是否适合检索
- 是否和你的模型服务集成方便

入门不要纠结最优模型。先选一个能生成 embedding 的模型，把流程跑通。

---

## 12. 常见问题

### Q：为什么关键词搜索不够？

关键词搜索看字面匹配。

向量检索看语义相似。

例如用户问：

```text
怎么让回答边生成边显示？
```

文档里写的是：

```text
使用 SSE 实现流式输出。
```

关键词可能匹配不到“边生成边显示”，但向量检索更容易找到“流式输出”。

### Q：向量检索一定比关键词搜索好吗？

不一定。

最好是混合检索：

```text
关键词检索 + 向量检索 + rerank
```

入门先掌握向量检索。

### Q：topK 越大越好吗？

不一定。topK 太大，会把无关内容塞进 prompt。

建议：

```text
先用 topK = 3 或 5
```

---

## 13. 跟学任务

- [ ] 写一个 `chunkText(text)` 函数
- [ ] 用 2 段文档手工构造 embedding 数组
- [ ] 写一个 cosine similarity 函数
- [ ] 写一个内存 search 函数
- [ ] 理解为什么 chunk 要存 source
- [ ] 解释关键词搜索和向量搜索的区别
- [ ] 画出 RAG 检索链路

---

## 下一步

```text
✅ 你已经理解 Embedding 和向量检索
✅ 你已经知道 RAG 为什么要先检索再回答

下一步 → 07_RAG文档问答.md
```
