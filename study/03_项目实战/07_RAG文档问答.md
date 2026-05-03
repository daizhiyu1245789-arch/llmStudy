# RAG 文档问答

> RAG = Retrieval-Augmented Generation，检索增强生成。它是大模型应用最常见、最实用的落地方向之一。

---

## 0. 学完这一篇你会什么？

你会学会：

- RAG 要解决什么问题
- RAG 的完整链路
- 文档问答系统应该怎么设计接口
- 怎么把检索结果拼进 Prompt
- 怎么要求模型只根据资料回答
- 怎么返回引用来源
- RAG 常见错误怎么排查

最终你要能做出：

```text
用户上传文档
  ↓
系统处理成知识库
  ↓
用户提问
  ↓
系统检索相关片段
  ↓
模型基于片段回答
  ↓
回答附带来源
```

---

## 1. 为什么需要 RAG？

普通大模型有几个问题：

- 不知道你的私有文档
- 知识可能过期
- 容易编造
- 回答没有引用来源
- 不能直接访问你的数据库或文件

比如你问：

```text
我们项目里的 Node Hugging Face 示例怎么启动？
```

如果模型没看过你的项目文档，它只能猜。

RAG 的做法是：

```text
先从你的文档里找相关内容
再把内容交给模型
让模型基于这些内容回答
```

---

## 2. RAG 的完整流程

分成两条链路。

### 2.1 入库链路

```text
上传文档
  ↓
解析文本
  ↓
清洗文本
  ↓
切块 chunk
  ↓
生成 embedding
  ↓
存入向量库
```

这条链路通常在文档上传或更新时运行。

### 2.2 问答链路

```text
用户问题
  ↓
生成问题 embedding
  ↓
向量检索 topK chunks
  ↓
拼接 RAG Prompt
  ↓
调用大模型
  ↓
返回答案 + 引用来源
```

这条链路在用户每次提问时运行。

---

## 3. RAG 系统的最小接口设计

Node 后端可以先设计 3 个接口：

```text
POST /api/documents
上传或写入文档

POST /api/documents/search
只检索，不生成回答

POST /api/rag/ask
检索 + 生成回答
```

为什么要有单独 search 接口？

因为调试 RAG 时，你要先知道“检索有没有找对资料”。

如果检索错了，模型再强也会答错。

---

## 4. 文档入库数据结构

文档：

```ts
type Document = {
  id: string;
  title: string;
  source: string;
  content: string;
  createdAt: string;
};
```

切块：

```ts
type DocumentChunk = {
  id: string;
  documentId: string;
  title: string;
  source: string;
  content: string;
  chunkIndex: number;
  embedding: number[];
};
```

检索结果：

```ts
type SearchResult = DocumentChunk & {
  score: number;
};
```

RAG 回答：

```ts
type RagAnswer = {
  answer: string;
  sources: Array<{
    title: string;
    source: string;
    chunkIndex: number;
    score: number;
  }>;
};
```

---

## 5. RAG Prompt 怎么写？

核心原则：

```text
模型只能根据给定资料回答。
资料没有就说不知道。
回答要带引用。
```

模板：

```text
你是一个严谨的文档问答助手。

任务：
请根据“参考资料”回答用户问题。

规则：
- 只能使用参考资料中的信息
- 如果参考资料没有答案，请回答“资料中没有找到相关信息”
- 不要编造
- 回答要简洁
- 关键结论后面标注来源编号，比如 [1]、[2]

参考资料：
{context}

用户问题：
{question}

输出：
```

context 可以这样拼：

```text
[1] 标题：Node 集成 Hugging Face
来源：study/03_项目实战/04_Node集成HuggingFace.md
内容：
进入 examples/node-huggingface 后运行 npm install，再配置 .env，最后 npm run dev。

[2] 标题：...
来源：...
内容：
...
```

---

## 6. Node 里构造 RAG Prompt

```ts
type RagContextItem = {
  title: string;
  source: string;
  content: string;
  score: number;
};

export function buildRagPrompt(question: string, contexts: RagContextItem[]) {
  const contextText = contexts
    .map((item, index) => {
      const sourceNo = index + 1;

      return [
        `[${sourceNo}] 标题：${item.title}`,
        `来源：${item.source}`,
        `相关度：${item.score.toFixed(3)}`,
        "内容：",
        item.content,
      ].join("\n");
    })
    .join("\n\n");

  return [
    "你是一个严谨的文档问答助手。",
    "",
    "任务：",
    "请根据“参考资料”回答用户问题。",
    "",
    "规则：",
    "- 只能使用参考资料中的信息",
    "- 如果参考资料没有答案，请回答“资料中没有找到相关信息”",
    "- 不要编造",
    "- 回答要简洁",
    "- 关键结论后面标注来源编号，比如 [1]、[2]",
    "",
    "参考资料：",
    contextText || "无",
    "",
    "用户问题：",
    question,
  ].join("\n");
}
```

---

## 7. RAG 问答伪代码

```ts
app.post("/api/rag/ask", async (req, res) => {
  const { question } = req.body;

  // 1. 生成问题向量
  const queryEmbedding = await embed(question);

  // 2. 检索相关 chunk
  const results = await vectorStore.search(queryEmbedding, { topK: 5 });

  // 3. 拼 prompt
  const prompt = buildRagPrompt(question, results);

  // 4. 调模型
  const answer = await generateText({
    prompt,
    maxNewTokens: 600,
    temperature: 0.2,
    topP: 0.9,
  });

  // 5. 返回答案和来源
  res.json({
    answer,
    sources: results.map((item, index) => ({
      ref: index + 1,
      title: item.title,
      source: item.source,
      chunkIndex: item.chunkIndex,
      score: item.score,
    })),
  });
});
```

注意 temperature 建议低一点：

```text
RAG 问答追求准确，不追求发散。
```

---

## 8. 引用来源怎么展示？

后端返回：

```json
{
  "answer": "启动方式是进入 examples/node-huggingface，安装依赖，配置 .env，然后运行 npm run dev。[1]",
  "sources": [
    {
      "ref": 1,
      "title": "Node 集成 Hugging Face",
      "source": "study/03_项目实战/04_Node集成HuggingFace.md",
      "chunkIndex": 3,
      "score": 0.82
    }
  ]
}
```

前端可以显示：

```text
回答：
启动方式是进入 examples/node-huggingface...

引用：
[1] Node 集成 Hugging Face
    study/03_项目实战/04_Node集成HuggingFace.md
```

引用来源很重要：

- 用户能检查答案
- 开发者能调试检索
- 避免模型看起来“凭空知道”

---

## 9. RAG 常见错误排查

### 9.1 检索没找对

表现：

```text
sources 和问题无关
```

排查：

- chunk 是否太长或太短
- embedding 模型是否适合中文
- topK 是否太小
- 文档是否真的入库
- query 是否被改写坏了

### 9.2 检索找对了，但模型答错

表现：

```text
sources 对，answer 错
```

排查：

- prompt 是否强调“只根据资料回答”
- context 是否太长
- temperature 是否太高
- 资料内部是否互相冲突
- 模型能力是否太弱

### 9.3 答案没有引用

排查：

- prompt 是否要求用 [1]、[2]
- context 是否编号
- 后端是否返回 sources
- 前端是否展示 sources

### 9.4 回答“资料中没有找到”太频繁

排查：

- 检索阈值是否太高
- topK 是否太小
- 文档切块是否丢了关键信息
- 用户问题是否需要改写

---

## 10. RAG 参数建议

| 参数 | 入门建议 |
|---|---|
| chunkSize | 500-1000 中文字符 |
| overlap | 80-150 中文字符 |
| topK | 3-5 |
| temperature | 0.1-0.4 |
| maxNewTokens | 400-800 |

不要一开始就追求复杂。先把这条链路跑通：

```text
小文档 → 切块 → 检索 → 回答 → 引用
```

---

## 11. RAG 和微调的区别

| 方式 | 适合 | 不适合 |
|---|---|---|
| RAG | 私有知识、频繁更新、需要引用 | 让模型学新语气/新风格 |
| 微调 | 固定任务格式、特定风格、分类抽取 | 频繁更新知识库 |

一句话：

```text
知识更新优先 RAG。
行为风格优先微调。
```

大多数业务应用先做 RAG，不要一上来就微调。

---

## 12. 跟学任务

- [ ] 写出 RAG 的入库链路
- [ ] 写出 RAG 的问答链路
- [ ] 设计 `DocumentChunk` 类型
- [ ] 写 `buildRagPrompt(question, contexts)`
- [ ] 让回答必须带 `[1]` 引用
- [ ] 单独实现一个 `/api/documents/search`
- [ ] 用 search 接口调试检索结果
- [ ] 解释“检索错”和“生成错”的区别

---

## 下一步

```text
✅ 你已经知道 RAG 文档问答怎么做
✅ 你已经知道引用来源和检索调试为什么重要

下一步 → 08_ToolCalling工具调用.md
```
