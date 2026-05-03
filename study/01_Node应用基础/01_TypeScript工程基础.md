# TypeScript 工程基础

> 大模型应用里，模型输出不稳定、用户输入不可控、接口参数复杂。TypeScript 和 Zod 是你的第一道防线。

---

## 0. 学完这一篇你会什么？

你会学会：

- 怎么用类型描述数据结构
- `type` 和 `interface` 怎么选
- 为什么 API 入参一定要校验
- 怎么用 Zod 校验请求体
- 怎么写更可靠的大模型调用参数
- 怎么给 RAG、Tool Calling 设计类型

---

## 1. 为什么大模型应用更需要 TypeScript？

普通接口里，返回格式通常是后端自己控制的。

大模型应用里，多了一个不稳定来源：

```text
用户输入不可控
模型输出不稳定
工具参数可能缺字段
RAG 检索结果结构复杂
```

所以你要尽早用类型把数据形状固定下来。

---

## 2. 用 type 描述模型调用参数

```ts
export type GenerateOptions = {
  prompt: string;
  maxNewTokens: number;
  temperature: number;
  topP: number;
};
```

这个类型表达：

```text
调用模型时，必须有 prompt、maxNewTokens、temperature、topP。
```

调用函数时就清楚很多：

```ts
async function generateText(options: GenerateOptions): Promise<string> {
  // 调 Hugging Face
  return "...";
}
```

---

## 3. type 和 interface 怎么选？

入门可以简单记：

| 场景 | 推荐 |
|---|---|
| 普通对象结构 | `type` 或 `interface` 都可以 |
| 联合类型 | `type` |
| 类实现契约 | `interface` |
| 业务 DTO | `type` |

例子：

```ts
type ModelName = "gpt2" | "qwen" | "llama";

type GenerateResult = {
  model: string;
  generatedText: string;
};
```

---

## 4. 可选字段和默认值

```ts
type GenerateRequest = {
  prompt: string;
  maxNewTokens?: number;
  temperature?: number;
};
```

`?` 表示字段可选。

但注意：TypeScript 只在编译期有用，运行时用户依然可能传乱七八糟的东西。

所以还需要 Zod。

---

## 5. 为什么需要 Zod？

TypeScript 不能阻止真实请求传错参数。

用户可能传：

```json
{
  "prompt": "",
  "maxNewTokens": "很多",
  "temperature": 999
}
```

所以后端要运行时校验。

安装：

```bash
npm install zod
```

定义 schema：

```ts
import { z } from "zod";

const GenerateSchema = z.object({
  prompt: z.string().trim().min(1).max(8000),
  maxNewTokens: z.coerce.number().int().min(1).max(512).default(120),
  temperature: z.coerce.number().min(0.1).max(2).default(0.8),
  topP: z.coerce.number().min(0.1).max(1).default(0.95),
});
```

使用：

```ts
const input = GenerateSchema.parse(req.body);
```

如果参数错了，Zod 会抛错，你可以统一返回 400。

---

## 6. 从 Zod 推导 TypeScript 类型

不要重复写两份类型。

```ts
type GenerateInput = z.infer<typeof GenerateSchema>;
```

这样 schema 改了，类型会自动跟着变。

```ts
async function generateText(input: GenerateInput) {
  // input 已经是校验后的类型
}
```

---

## 7. 给 RAG 设计类型

```ts
export type DocumentChunk = {
  id: string;
  documentId: string;
  title: string;
  source: string;
  content: string;
  chunkIndex: number;
  embedding: number[];
};

export type SearchResult = DocumentChunk & {
  score: number;
};
```

好处：

- 检索结果里一定有 `score`
- 前端展示引用时知道有哪些字段
- 日志记录时结构统一

---

## 8. 给 Tool Calling 设计类型

```ts
type ToolCall = {
  type: "tool_call";
  toolName: string;
  arguments: unknown;
};

type FinalAnswer = {
  type: "final";
  answer: string;
};

type ModelAction = ToolCall | FinalAnswer;
```

这叫联合类型。

使用时可以自动缩小类型：

```ts
function handleAction(action: ModelAction) {
  if (action.type === "tool_call") {
    return runTool(action.toolName, action.arguments);
  }

  return action.answer;
}
```

---

## 9. 常见类型习惯

### 9.1 不要到处用 any

不好：

```ts
function runTool(args: any) {}
```

更好：

```ts
function runTool(args: unknown) {
  const parsed = ToolArgsSchema.parse(args);
}
```

`unknown` 表示：我不知道它是什么，必须校验后才能用。

### 9.2 API 返回也要有类型

```ts
type GenerateResponse = {
  model: string;
  generatedText: string;
};
```

### 9.3 日志也要有类型

```ts
type LlmLog = {
  requestId: string;
  model: string;
  latencyMs: number;
  status: "success" | "error";
};
```

---

## 10. 跟学任务

- [ ] 给 `/api/generate` 请求体写 Zod schema
- [ ] 用 `z.infer` 推导请求类型
- [ ] 给 RAG chunk 写 `DocumentChunk` 类型
- [ ] 给 Tool Calling 写 `ModelAction` 联合类型
- [ ] 把 `any` 改成 `unknown + Zod parse`
- [ ] 给模型返回结果写类型

---

## 下一步

```text
✅ 你已经知道 TypeScript 在大模型应用里怎么保护边界
✅ 你已经知道 Zod 是运行时校验，不是类型装饰

下一步 → 02_HTTP接口和SSE流式响应.md
```
