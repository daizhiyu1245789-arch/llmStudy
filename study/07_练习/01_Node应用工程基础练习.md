# Node 应用工程基础练习

> 对应 `study/01_Node应用工程基础/`。这些练习是后面 Hugging Face、RAG、Tool Calling 的工程地基。

---

## 01_TypeScript 工程基础

对应章节：

```text
01_Node应用工程基础/01_TypeScript工程基础.md
```

### 基础练习

- [ ] 写出 `GenerateOptions` 类型，包含 `prompt`、`maxNewTokens`、`temperature`、`topP`
- [ ] 写出 `GenerateResult` 类型，包含 `model`、`generatedText`
- [ ] 解释 `type` 和 `interface` 的区别
- [ ] 解释为什么大模型应用不建议到处用 `any`
- [ ] 解释 `unknown + Zod parse` 为什么更安全

### 实战练习

写一个 Zod schema：

```ts
const GenerateSchema = z.object({
  prompt: z.string().trim().min(1).max(8000),
  maxNewTokens: z.coerce.number().int().min(1).max(512).default(120),
  temperature: z.coerce.number().min(0.1).max(2).default(0.8),
  topP: z.coerce.number().min(0.1).max(1).default(0.95),
});
```

然后完成：

- [ ] 用 `z.infer` 推导 `GenerateInput`
- [ ] 写一个函数 `normalizeGenerateInput(raw: unknown): GenerateInput`
- [ ] 传入错误参数，观察 Zod 报错

### 验收标准

你能回答：

```text
TypeScript 是编译期保护，Zod 是运行时保护。
```

---

## 02_HTTP 接口和 SSE 流式响应

对应章节：

```text
01_Node应用工程基础/02_HTTP接口和SSE流式响应.md
```

### 基础练习

- [ ] 解释普通 HTTP API 和 SSE 的区别
- [ ] 写出 SSE 必须设置的 3 个响应头
- [ ] 写出 SSE 数据格式：`data: ...\n\n`
- [ ] 解释为什么聊天应用适合 SSE

### 实战练习

实现一个假流式接口：

```text
POST /api/fake-stream
```

每 500ms 输出一段：

```text
Hello
 from
 SSE
```

要求：

- [ ] 后端用 `res.write`
- [ ] 前端用 `fetch + reader`
- [ ] 页面能边接收边显示
- [ ] 最后发送 `{ done: true }`

### 验收标准

你能说清：

```text
SSE 不是一次性返回 JSON，而是服务端多次写入响应体。
```

---

## 03_环境变量和密钥安全

对应章节：

```text
01_Node应用工程基础/03_环境变量和密钥安全.md
```

### 基础练习

- [ ] 写 `.env.example`
- [ ] 写 `.gitignore` 忽略 `.env`
- [ ] 解释为什么 `HF_TOKEN` 不能放前端
- [ ] 解释为什么不能把 token 写死在代码里
- [ ] 列出大模型应用里的 5 类敏感信息

### 实战练习

写 `config.ts`：

```ts
export const config = {
  port: 3001,
  hfToken: process.env.HF_TOKEN,
  hfModel: process.env.HF_MODEL ?? "gpt2",
};
```

然后补齐：

- [ ] `optionalEnv(name)`
- [ ] `numberEnv(name, fallback)`
- [ ] `assertHuggingFaceConfig()`
- [ ] 缺少 `HF_TOKEN` 时启动报错

### 验收标准

你能解释：

```text
前端环境变量只要进入打包产物，就不再是秘密。
```

---

## 04_异步编程和错误处理

对应章节：

```text
01_Node应用工程基础/04_异步编程和错误处理.md
```

### 基础练习

- [ ] 解释 `async` 函数返回什么
- [ ] 解释 `await` 出错后会发生什么
- [ ] 解释 `try/catch/next(error)` 在 Express 里的作用
- [ ] 区分 400 参数错误和 500 服务错误

### 实战练习

完成：

- [ ] 给 `/api/generate` 加 `try/catch`
- [ ] 写统一错误处理中间件
- [ ] Zod 错误返回 400
- [ ] 普通错误返回 500
- [ ] 每个错误响应带 `requestId`
- [ ] 写 `withTimeout(promise, ms)`

### 加分挑战

- [ ] 模拟模型请求超时
- [ ] 网络错误重试 1 次
- [ ] 参数错误不重试

### 验收标准

你能说清：

```text
不是所有错误都应该重试，重试会增加成本。
```

---

## 05_文件处理和文本预处理

对应章节：

```text
01_Node应用工程基础/05_文件处理和文本预处理.md
```

### 基础练习

- [ ] 读取一个 `.md` 文件
- [ ] 实现 `normalizeText(text)`
- [ ] 实现 `chunkText(text, chunkSize, overlap)`
- [ ] 实现 `splitMarkdownByHeading(markdown)`
- [ ] 给每个 chunk 加 `source`、`chunkIndex`

### 实战练习

读取项目根目录 `README.md`：

- [ ] 清洗文本
- [ ] 按 800 字符切块
- [ ] overlap 设置为 120
- [ ] 打印每个 chunk 的 index 和前 80 个字符

### 加分挑战

- [ ] 先按 Markdown 标题切
- [ ] 如果某块超过 1000 字符，再二次切块

### 验收标准

你能解释：

```text
RAG 的质量很大程度取决于文档切块质量。
```

---

## 06_Node 大模型应用项目结构

对应章节：

```text
01_Node应用工程基础/06_Node大模型应用项目结构.md
```

### 基础练习

- [ ] 解释 `routes` 放什么
- [ ] 解释 `services` 放什么
- [ ] 解释 `providers` 放什么
- [ ] 解释 `prompts` 放什么
- [ ] 解释 `stores` 放什么
- [ ] 解释 `tools` 放什么

### 实战练习

把一个单文件 `/api/generate` 拆成：

```text
routes/generate.route.ts
services/generation.service.ts
providers/huggingface.provider.ts
schemas/generate.schema.ts
```

要求：

- [ ] route 只处理 HTTP
- [ ] service 编排业务流程
- [ ] provider 封装 Hugging Face
- [ ] schema 做参数校验

### 验收标准

你能画出：

```text
request → route → service → provider → model → service → route → response
```
