# Node 大模型应用项目结构

> 大模型 demo 可以写在一个文件里，但真实项目最好从一开始就分层。分层清楚，后面加 Prompt、RAG、Tool Calling、日志和部署才不会乱。

---

## 0. 学完这一篇你会什么？

你会学会：

- 一个 Node 大模型项目应该怎么分目录
- route / service / provider / store 分别放什么
- Prompt 模板放哪里
- RAG 逻辑放哪里
- Tool Calling 逻辑放哪里
- 日志和配置怎么组织
- 怎么从 demo 慢慢演进成真实项目

---

## 1. 为什么要分层？

最小 demo 可能这样写：

```ts
app.post("/api/generate", async (req, res) => {
  const client = new InferenceClient(process.env.HF_TOKEN);
  const output = await client.textGeneration({
    model: "gpt2",
    inputs: req.body.prompt,
  });
  res.json(output);
});
```

能跑，但很快会变乱：

- 参数校验混在接口里
- Prompt 拼接混在接口里
- 模型 SDK 混在业务逻辑里
- RAG 检索混在生成接口里
- 工具调用没有边界
- 日志到处 `console.log`

分层的目标：

```text
每个文件只负责一类事情。
```

---

## 2. 推荐目录结构

```text
src/
├── server.ts
├── config.ts
├── routes/
│   ├── generate.route.ts
│   ├── rag.route.ts
│   └── tools.route.ts
├── services/
│   ├── generation.service.ts
│   ├── rag.service.ts
│   └── tool-agent.service.ts
├── providers/
│   ├── huggingface.provider.ts
│   └── embedding.provider.ts
├── prompts/
│   ├── teacher.prompt.ts
│   ├── rag.prompt.ts
│   └── tool-router.prompt.ts
├── stores/
│   ├── vector.store.ts
│   └── document.store.ts
├── tools/
│   ├── registry.ts
│   └── get-weather.tool.ts
├── schemas/
│   ├── generate.schema.ts
│   ├── rag.schema.ts
│   └── tool.schema.ts
├── utils/
│   ├── logger.ts
│   ├── request-id.ts
│   ├── timeout.ts
│   └── text.ts
└── types/
    ├── rag.ts
    └── tool.ts
```

---

## 3. 每一层负责什么？

### 3.1 routes

只负责 HTTP：

- 读取请求
- 参数校验
- 调 service
- 返回响应

例子：

```ts
router.post("/generate", async (req, res, next) => {
  try {
    const input = GenerateSchema.parse(req.body);
    const result = await generationService.generate(input);
    res.json(result);
  } catch (error) {
    next(error);
  }
});
```

不要在 route 里直接拼 prompt、查向量库、调模型。

### 3.2 services

负责业务流程：

```text
生成服务：拼 prompt → 调模型 → 返回结果
RAG 服务：生成问题向量 → 检索 → 拼 RAG prompt → 调模型
Agent 服务：让模型选工具 → 执行工具 → 总结结果
```

### 3.3 providers

封装第三方服务：

```text
Hugging Face Provider
Embedding Provider
向量数据库 Provider
```

好处：

以后你从 Hugging Face 换到别的模型，不用改业务层。

### 3.4 prompts

只放 Prompt 模板。

```ts
export function buildRagPrompt(question: string, contexts: RagContext[]) {
  return [...].join("\n");
}
```

Prompt 要有版本号：

```ts
export const RAG_PROMPT_VERSION = "rag-v1";
```

### 3.5 stores

负责数据存取：

```text
document.store.ts：保存文档
vector.store.ts：保存和检索向量
```

学习阶段可以先用内存数组。

以后再替换成：

```text
Postgres + pgvector
Qdrant
SQLite
```

### 3.6 tools

放工具定义和执行函数：

```text
get-weather.tool.ts
search-docs.tool.ts
calculate.tool.ts
registry.ts
```

工具要有：

- name
- description
- schema
- handler

### 3.7 schemas

放 Zod schema：

```ts
export const GenerateSchema = z.object({
  prompt: z.string().min(1),
});
```

### 3.8 utils

放通用工具：

- logger
- timeout
- requestId
- text normalize
- hash

---

## 4. 一个请求的完整流动

以 RAG 问答为例：

```text
POST /api/rag/ask
  ↓
rag.route.ts 校验参数
  ↓
rag.service.ts 编排流程
  ↓
embedding.provider.ts 生成问题向量
  ↓
vector.store.ts 检索相关 chunks
  ↓
rag.prompt.ts 拼 Prompt
  ↓
huggingface.provider.ts 调模型
  ↓
rag.service.ts 组装 answer + sources
  ↓
route 返回 JSON
```

这条链路清楚，后面排查问题就容易。

---

## 5. 从 demo 演进到真实项目

### 第 1 版：单文件

```text
server.ts
```

适合验证 SDK 能不能跑。

### 第 2 版：拆 provider

```text
server.ts
huggingface.ts
config.ts
```

适合封装模型调用。

### 第 3 版：拆 route / service

```text
routes/
services/
providers/
```

适合开始写业务功能。

### 第 4 版：加 RAG 和 tools

```text
prompts/
stores/
tools/
schemas/
```

适合真实应用。

### 第 5 版：加日志部署

```text
utils/logger.ts
evals/
Dockerfile
```

适合上线。

---

## 6. 命名建议

| 类型 | 命名 |
|---|---|
| 路由 | `xxx.route.ts` |
| 业务服务 | `xxx.service.ts` |
| 第三方封装 | `xxx.provider.ts` |
| Prompt | `xxx.prompt.ts` |
| Zod schema | `xxx.schema.ts` |
| 工具 | `xxx.tool.ts` |
| 数据存储 | `xxx.store.ts` |

统一命名比“随手起名”重要很多。

---

## 7. 跟学任务

- [ ] 画出你自己的 Node 大模型项目目录
- [ ] 把模型调用放进 provider
- [ ] 把 HTTP 接口放进 route
- [ ] 把业务流程放进 service
- [ ] 把 prompt 模板放进 prompts
- [ ] 把 Zod schema 放进 schemas
- [ ] 解释 RAG 请求从 route 到 provider 的完整链路

---

## 下一步

```text
✅ 你已经知道大模型 Node 项目怎么分层
✅ 后面做 RAG、Tool Calling、部署时不会堆成一个巨大的 server.ts
```
