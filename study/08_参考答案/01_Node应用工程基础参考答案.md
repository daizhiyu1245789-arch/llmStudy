# Node 应用工程基础参考答案

> 对应 `07_练习/01_Node应用工程基础练习.md`。

---

## 01_TypeScript 工程基础

### GenerateOptions

```ts
export type GenerateOptions = {
  prompt: string;
  maxNewTokens: number;
  temperature: number;
  topP: number;
};
```

### GenerateResult

```ts
export type GenerateResult = {
  model: string;
  generatedText: string;
};
```

### Zod Schema

```ts
import { z } from "zod";

export const GenerateSchema = z.object({
  prompt: z.string().trim().min(1, "prompt is required").max(8000),
  maxNewTokens: z.coerce.number().int().min(1).max(512).default(120),
  temperature: z.coerce.number().min(0.1).max(2).default(0.8),
  topP: z.coerce.number().min(0.1).max(1).default(0.95),
});

export type GenerateInput = z.infer<typeof GenerateSchema>;

export function normalizeGenerateInput(raw: unknown): GenerateInput {
  return GenerateSchema.parse(raw);
}
```

### 为什么 unknown 比 any 更安全？

```text
any：绕过类型检查，直接随便用。
unknown：你必须先判断或校验，才能使用。
```

在大模型应用里，用户输入和模型输出都不可信，所以推荐：

```text
unknown → Zod parse → 可信类型
```

---

## 02_HTTP 接口和 SSE 流式响应

### SSE 响应头

```ts
res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
res.setHeader("Cache-Control", "no-cache, no-transform");
res.setHeader("Connection", "keep-alive");
res.flushHeaders();
```

### 假流式接口

```ts
app.post("/api/fake-stream", async (_req, res) => {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  const tokens = ["Hello", " from", " SSE"];

  for (const token of tokens) {
    res.write(`data: ${JSON.stringify({ token })}\n\n`);
    await sleep(500);
  }

  res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
  res.end();
});

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
```

### 前端读取

```js
const response = await fetch("/api/fake-stream", { method: "POST" });
const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { value, done } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const events = buffer.split("\n\n");
  buffer = events.pop() || "";

  for (const event of events) {
    const line = event.split("\n").find((item) => item.startsWith("data: "));
    if (!line) continue;

    const data = JSON.parse(line.slice(6));
    if (data.token) {
      console.log(data.token);
    }
  }
}
```

---

## 03_环境变量和密钥安全

### `.env.example`

```bash
HF_TOKEN=hf_your_token_here
HF_MODEL=gpt2
PORT=3001
CORS_ORIGIN=http://localhost:3001
```

### `.gitignore`

```text
.env
node_modules/
dist/
.DS_Store
```

### config.ts

```ts
import "dotenv/config";

function optionalEnv(name: string): string | undefined {
  const value = process.env[name]?.trim();
  return value ? value : undefined;
}

function numberEnv(name: string, fallback: number): number {
  const raw = optionalEnv(name);
  if (!raw) return fallback;

  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be a number`);
  }

  return value;
}

export const config = {
  port: numberEnv("PORT", 3001),
  hfToken: optionalEnv("HF_TOKEN"),
  hfModel: optionalEnv("HF_MODEL") ?? "gpt2",
};

export function assertHuggingFaceConfig() {
  if (!config.hfToken) {
    throw new Error("Missing HF_TOKEN");
  }
}
```

### 为什么 `VITE_HF_TOKEN` 不安全？

```text
VITE_ 开头的变量会进入前端打包产物。
只要进入浏览器，就不能当作秘密。
```

---

## 04_异步编程和错误处理

### Express 接口错误处理

```ts
app.post("/api/generate", async (req, res, next) => {
  const requestId = crypto.randomUUID();

  try {
    const input = GenerateSchema.parse(req.body);
    const result = await withTimeout(generateText(input), 60_000);

    res.json({
      requestId,
      result,
    });
  } catch (error) {
    (req as any).requestId = requestId;
    next(error);
  }
});
```

### 错误中间件

```ts
app.use((error: unknown, req: Request, res: Response, _next: NextFunction) => {
  const requestId = (req as any).requestId ?? crypto.randomUUID();

  if (error instanceof z.ZodError) {
    res.status(400).json({
      requestId,
      error: "Invalid request",
      issues: error.issues,
    });
    return;
  }

  console.error({ requestId, error });

  res.status(500).json({
    requestId,
    error: "Internal server error",
  });
});
```

### withTimeout

```ts
export function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Timeout after ${ms}ms`));
    }, ms);

    promise
      .then(resolve)
      .catch(reject)
      .finally(() => clearTimeout(timer));
  });
}
```

---

## 05_文件处理和文本预处理

### readTextFile

```ts
import fs from "node:fs/promises";

export async function readTextFile(filePath: string): Promise<string> {
  return fs.readFile(filePath, "utf8");
}
```

### normalizeText

```ts
export function normalizeText(text: string): string {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]+/g, " ")
    .trim();
}
```

### chunkText

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

### splitMarkdownByHeading

```ts
export function splitMarkdownByHeading(markdown: string): string[] {
  return markdown
    .split(/\n(?=##\s+)/)
    .map((part) => part.trim())
    .filter(Boolean);
}
```

### chunk 元数据

```ts
type Chunk = {
  id: string;
  documentId: string;
  title: string;
  source: string;
  content: string;
  chunkIndex: number;
};
```

---

## 06_Node 大模型应用项目结构

### 推荐分层

```text
request
  ↓
route：处理 HTTP
  ↓
schema：校验参数
  ↓
service：编排业务流程
  ↓
provider：调用外部模型/服务
  ↓
response
```

### generate 示例拆分

```text
routes/generate.route.ts
services/generation.service.ts
providers/huggingface.provider.ts
schemas/generate.schema.ts
```

参考职责：

- `generate.route.ts`：接收请求，返回响应
- `generation.service.ts`：拼 prompt，调用 provider
- `huggingface.provider.ts`：只封装 Hugging Face SDK
- `generate.schema.ts`：只放 Zod schema
