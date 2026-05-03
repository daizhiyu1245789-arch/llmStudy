# Node 集成 Hugging Face

> 这一篇按“跟着做就能跑起来”的方式写。你会从 0 创建一个 Node 后端，接入 Hugging Face，提供普通生成和流式生成接口，最后用浏览器页面联调。

---

## 0. 学完这一篇你会什么？

你会完成一个小服务：

```text
浏览器页面
  ↓
Node / Express API
  ↓
Hugging Face 模型
  ↓
返回生成结果
```

你会掌握：

- 如何在 Node 后端保存 Hugging Face Token
- 如何调用文本生成模型
- 如何设计 /api/generate
- 如何设计 /api/generate/stream
- 如何用 SSE 做流式输出
- 如何用前端页面读取 stream
- 如何处理参数校验和错误

---

## 1. 本示例做什么？

代码位置：

```text
examples/node-huggingface/
```

功能：

- GET /health：健康检查
- POST /api/generate：一次性返回完整生成结果
- POST /api/generate/stream：用 SSE 流式返回 token
- GET /：一个简单浏览器页面，用来测试接口

技术栈：

```text
Node.js + TypeScript + Express + @huggingface/inference + Zod
```

---

## 2. 先理解整体链路

### 2.1 普通生成链路

```text
用户输入 prompt
  ↓
浏览器 POST /api/generate
  ↓
Node 读取 prompt
  ↓
Node 调 Hugging Face
  ↓
Hugging Face 一次性返回完整文本
  ↓
Node 返回 JSON
  ↓
浏览器显示 generatedText
```

普通生成的特点：

- 代码简单
- 适合短文本
- 用户必须等模型完整生成后才能看到结果

### 2.2 流式生成链路

```text
用户输入 prompt
  ↓
浏览器 POST /api/generate/stream
  ↓
Node 调 Hugging Face stream
  ↓
模型每生成一点就返回一个 chunk
  ↓
Node 用 SSE 推给浏览器
  ↓
浏览器边接收边显示
```

流式生成的特点：

- 用户体感更快
- 适合聊天产品
- 前后端都要处理 stream
- 错误处理比普通接口复杂一点

普通生成像“整篇写完再发给你”。流式生成像“边写边给你看”。

---

## 3. 为什么 token 必须放后端？

Hugging Face token 不应该放在浏览器里。

正确结构：

```text
浏览器
  ↓ 调你自己的 Node API
Node 服务
  ↓ 带 HF_TOKEN 调 Hugging Face
Hugging Face 模型服务
```

这样可以隐藏 token、做用户鉴权、限流、日志、成本统计，也方便以后接数据库、RAG、工具调用。

错误做法：

```text
浏览器直接带 HF_TOKEN 请求 Hugging Face
```

这样 token 会暴露给用户。

---

## 4. 准备 Hugging Face Token

步骤：

```text
1. 打开 https://huggingface.co
2. 登录账号
3. 进入 Settings
4. 找到 Access Tokens
5. 创建一个 token
6. 复制 token，形如 hf_xxx
```

本地开发时，把 token 放到 .env：

```bash
HF_TOKEN=hf_your_token_here
```

不要提交 .env 到 Git。

---

## 5. 安装和运行

进入示例目录：

```bash
cd examples/node-huggingface
npm install
cp .env.example .env
```

编辑 .env：

```bash
HF_TOKEN=hf_your_token_here
HF_MODEL=gpt2
PORT=3001
```

启动：

```bash
npm run dev
```

打开：

```bash
http://localhost:3001
```

---

## 6. 项目文件说明

```text
examples/node-huggingface/
├── package.json
├── tsconfig.json
├── .env.example
├── README.md
├── src/
│   ├── config.ts
│   ├── huggingface.ts
│   └── server.ts
└── public/
    └── index.html
```

| 文件 | 作用 |
|---|---|
| package.json | npm 依赖和启动命令 |
| .env.example | 环境变量模板 |
| src/config.ts | 读取环境变量 |
| src/huggingface.ts | 封装 Hugging Face 调用 |
| src/server.ts | Express API 服务 |
| public/index.html | 测试页面 |

---

## 7. 第一步：环境变量配置

文件：src/config.ts

核心代码：

```ts
import 'dotenv/config';

export const config = {
  port: numberEnv('PORT', 3001),
  corsOrigin: optionalEnv('CORS_ORIGIN') ?? 'http://localhost:3001',
  hfToken: optionalEnv('HF_TOKEN'),
  hfModel: optionalEnv('HF_MODEL') ?? 'gpt2',
  hfEndpointUrl: optionalEnv('HF_ENDPOINT_URL'),
};
```

这段代码做了几件事：读取 .env、设置默认端口 3001、设置默认模型 gpt2，并支持可选的 Hugging Face Inference Endpoint。

为什么要有 assertHuggingFaceConfig？

```ts
export function assertHuggingFaceConfig(): void {
  if (!config.hfToken) {
    throw new Error('Missing HF_TOKEN. Create .env from .env.example and set your Hugging Face token.');
  }
}
```

因为 HF_TOKEN 是必填的。提前报错比接口调用时才失败更好排查。

---

## 8. 第二步：封装 Hugging Face 调用

文件：src/huggingface.ts

### 8.1 创建客户端

```ts
import { InferenceClient } from '@huggingface/inference';

const client = new InferenceClient(config.hfToken);
```

你可以把它理解成：

```ts
const client = new HuggingFaceSDK(token);
```

之后所有模型调用都通过 client 发出去。

### 8.2 普通生成

```ts
export async function generateText(options: GenerateOptions): Promise<string> {
  const output = await client.textGeneration({
    model: config.hfModel,
    inputs: options.prompt,
    parameters: {
      max_new_tokens: options.maxNewTokens,
      temperature: options.temperature,
      top_p: options.topP,
      return_full_text: false,
    },
  });

  return output.generated_text;
}
```

参数解释：

| 参数 | 含义 |
|---|---|
| model | 使用哪个 Hugging Face 模型 |
| inputs | 用户输入的 prompt |
| max_new_tokens | 最多生成多少新 token |
| temperature | 随机性，越大越发散 |
| top_p | 只从累计概率前 p 的候选里采样 |
| return_full_text | 是否把原 prompt 也返回 |

### 8.3 流式生成

```ts
export async function* streamText(options: GenerateOptions): AsyncGenerator<string> {
  const stream = client.textGenerationStream({
    model: config.hfModel,
    inputs: options.prompt,
    parameters: {
      max_new_tokens: options.maxNewTokens,
      temperature: options.temperature,
      top_p: options.topP,
      return_full_text: false,
    },
  });

  for await (const chunk of stream) {
    const token = chunk.token?.text;
    if (token) {
      yield token;
    }
  }
}
```

这里用了 async generator。模型每返回一个 token，就 yield 给 server.ts，server.ts 再立刻写给浏览器。

---

## 9. 第三步：写 Express 接口

文件：src/server.ts

### 9.1 参数校验

```ts
const generateSchema = z.object({
  prompt: z.string().trim().min(1, 'prompt is required').max(8000),
  maxNewTokens: z.coerce.number().int().min(1).max(512).default(120),
  temperature: z.coerce.number().min(0.1).max(2).default(0.8),
  topP: z.coerce.number().min(0.1).max(1).default(0.95),
});
```

为什么要校验？防止空 prompt、防止一次生成太长、防止 temperature 乱传，也方便前端调错时快速定位。

### 9.2 普通接口

```ts
app.post('/api/generate', async (req, res, next) => {
  try {
    const input = generateSchema.parse(req.body);
    const generatedText = await generateText(input);

    res.json({
      model: config.hfModel,
      generatedText,
    });
  } catch (error) {
    next(error);
  }
});
```

这就是标准后端接口：接收参数、校验、调模型、返回 JSON。

### 9.3 流式接口

```ts
app.post('/api/generate/stream', async (req, res, next) => {
  const input = generateSchema.parse(req.body);

  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  for await (const token of streamText(input)) {
    res.write('data: ' + JSON.stringify({ token }) + '\n\n');
  }

  res.write('data: ' + JSON.stringify({ done: true }) + '\n\n');
  res.end();
});
```

SSE 的格式必须是：

```text
data: 一段 JSON


data: 下一段 JSON

```

关键是每个事件后面要有空行，也就是 \n\n。

---

## 10. 第四步：前端读取流

文件：public/index.html

普通生成很简单：

```js
const response = await fetch('/api/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload()),
});
const data = await response.json();
outputEl.textContent = data.generatedText;
```

流式生成要读 response.body：

```js
const response = await fetch('/api/generate/stream', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload()),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
```

然后循环读取：

```js
while (true) {
  const { value, done } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  const events = buffer.split('\n\n');
  buffer = events.pop() || '';

  for (const event of events) {
    const line = event.split('\n').find((item) => item.startsWith('data: '));
    if (!line) continue;

    const data = JSON.parse(line.slice(6));
    if (data.token) outputEl.textContent += data.token;
  }
}
```

实际项目里，你可以把这段封装成 streamGenerate(payload, onToken)，React、Vue、原生页面都能复用。

---

## 11. 接口测试

### 11.1 健康检查

```bash
curl http://localhost:3001/health
```

### 11.2 普通生成

```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Deep learning is",
    "maxNewTokens": 80,
    "temperature": 0.8,
    "topP": 0.95
  }'
```

### 11.3 流式生成

```bash
curl -N -X POST http://localhost:3001/api/generate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is deep learning?",
    "maxNewTokens": 80
  }'
```

返回类似：

```text
data: {"token":"Deep"}

data: {"token":" learning"}

data: {"done":true}
```

---

## 12. 你可以怎么改？

### 12.1 换模型

改 .env：

```bash
HF_MODEL=gpt2
```

gpt2 主要适合英文和接口验证。如果你要中文，应该换中文能力更好的模型。

### 12.2 改默认生成长度

改 src/server.ts：

```ts
maxNewTokens: z.coerce.number().int().min(1).max(512).default(120)
```

### 12.3 加一个 prompt 模板

在 src/huggingface.ts 里调用前拼 prompt：

```ts
const finalPrompt = '你是一个耐心的编程老师。\n\n用户问题：' + options.prompt + '\n\n回答：';
```

然后把 finalPrompt 传给 inputs。

### 12.4 加日志

在 /api/generate 里记录耗时：

```ts
const startedAt = Date.now();
const generatedText = await generateText(input);
console.log({
  model: config.hfModel,
  latencyMs: Date.now() - startedAt,
  promptLength: input.prompt.length,
});
```

---

## 13. 常见问题

### Q：为什么用 gpt2？

因为它公开、轻量、容易测试。真实应用可以换成更强的模型或更适合业务的模型服务。

### Q：中文效果不好怎么办？

换支持中文更好的模型。gpt2 主要是英文模型，只适合验证接口链路。

### Q：为什么流式接口有时不可用？

流式输出依赖模型服务后端能力。Hugging Face 的 Text Generation Inference 支持 token streaming，但不同模型和 provider 的能力可能不同。

### Q：生产环境还缺什么？

至少还要补：用户鉴权、请求限流、日志、超时和重试、内容安全策略、token 成本统计、prompt 版本管理。

---

## 14. 跟学任务

学完这篇，你应该亲手完成：

- [ ] 跑起 npm run dev
- [ ] 打开页面，点一次普通生成
- [ ] 点一次流式生成
- [ ] 用 curl 调一次 /api/generate
- [ ] 用 curl 调一次 /api/generate/stream
- [ ] 把 maxNewTokens 改成 50 试试
- [ ] 把 temperature 改成 0.3 和 1.2 对比输出
- [ ] 改一次 prompt 模板
- [ ] 故意删掉 HF_TOKEN，观察错误
- [ ] 看懂 src/server.ts 的两个接口

如果这些你都能做完，这一章就过关了。

---

## 15. 官方资料

- Hugging Face JS Inference：https://huggingface.co/docs/huggingface.js/inference/modules
- Text Generation Streaming：https://huggingface.co/docs/text-generation-inference/en/conceptual/streaming
- Transformers.js：https://huggingface.co/docs/transformers.js

---

## 下一步

```text
✅ 你已经有了 Node 集成 Hugging Face 的完整服务端示例
✅ 可以继续加 RAG、工具调用、用户系统和日志
```
