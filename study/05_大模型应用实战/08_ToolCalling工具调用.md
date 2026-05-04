# Tool Calling 工具调用

> Tool Calling 让模型不只是回答文字，还能选择调用你的后端函数。它是工程 Agent 的核心能力之一。

---

## 0. 学完这一篇你会什么？

你会学会：

- Tool Calling 要解决什么问题
- 工具和普通 API 的区别
- 工具 schema 怎么设计
- 模型如何选择工具
- 后端如何校验和执行工具
- 工具调用的安全边界
- 一个最小 Agent 循环怎么写

---

## 1. 为什么需要工具调用？

普通模型只能“说”：

```text
用户：帮我查订单 123 的物流。
模型：你可以去订单系统查询。
```

接了工具后，模型可以“决定调用后端函数”：

```text
用户：帮我查订单 123 的物流。
模型：需要调用 getOrderShipping({ orderId: "123" })
后端：执行函数，拿到物流状态
模型：订单 123 已发出，当前到达上海转运中心。
```

工具调用适合：

- 查数据库
- 查订单
- 查天气
- 发起搜索
- 读取知识库
- 调内部业务 API
- 执行计算
- 生成报表

---

## 2. Tool Calling 的基本结构

```text
用户问题
  ↓
模型判断是否需要工具
  ↓
如果需要，输出工具名和参数
  ↓
后端校验参数和权限
  ↓
后端执行真实函数
  ↓
把工具结果交回模型
  ↓
模型生成最终回答
```

关键点：

```text
模型只负责“选择工具和生成参数”
后端负责“校验、权限、执行、兜底”
```

不要让模型直接决定危险操作。

---

## 3. 工具 schema 是什么？

工具 schema 描述一个工具：

```ts
type ToolDefinition = {
  name: string;
  description: string;
  parameters: object;
};
```

例子：

```ts
const getWeatherTool = {
  name: "getWeather",
  description: "查询指定城市的天气",
  parameters: {
    type: "object",
    properties: {
      city: {
        type: "string",
        description: "城市名称，比如 北京、上海、杭州",
      },
    },
    required: ["city"],
  },
};
```

模型看到这个描述后，才知道：

```text
有一个工具叫 getWeather
它能查天气
它需要 city 参数
```

---

## 4. 工具函数怎么写？

真实执行逻辑在后端。

```ts
type GetWeatherArgs = {
  city: string;
};

export async function getWeather(args: GetWeatherArgs) {
  return {
    city: args.city,
    weather: "晴",
    temperature: "26°C",
  };
}
```

注意：

```text
工具函数不应该相信模型传入的参数。
```

一定要校验。

---

## 5. 用 Zod 校验工具参数

```ts
import { z } from "zod";

const GetWeatherArgsSchema = z.object({
  city: z.string().min(1).max(50),
});

export async function runGetWeather(rawArgs: unknown) {
  const args = GetWeatherArgsSchema.parse(rawArgs);

  return getWeather(args);
}
```

为什么要校验？

- 模型可能漏参数
- 模型可能传错类型
- 用户可能诱导模型传危险参数
- 工具可能涉及数据库、文件、外部 API

---

## 6. 最小工具注册表

```ts
type ToolHandler = (args: unknown) => Promise<unknown>;

const toolRegistry: Record<string, ToolHandler> = {
  getWeather: runGetWeather,
};

export async function runTool(name: string, args: unknown) {
  const handler = toolRegistry[name];

  if (!handler) {
    throw new Error(`Unknown tool: ${name}`);
  }

  return handler(args);
}
```

这样可以避免模型调用不存在的函数。

---

## 7. 如果模型不支持原生 Tool Calling 怎么办？

有些模型 API 支持原生 tool calling，有些不支持。

如果不支持，可以用 JSON 协议模拟。

Prompt：

```text
你可以选择调用工具，也可以直接回答。

可用工具：
1. getWeather(city: string)：查询城市天气

如果需要调用工具，只输出 JSON：
{
  "type": "tool_call",
  "toolName": "getWeather",
  "arguments": {
    "city": "杭州"
  }
}

如果不需要调用工具，只输出 JSON：
{
  "type": "final",
  "answer": "..."
}
```

模型输出：

```json
{
  "type": "tool_call",
  "toolName": "getWeather",
  "arguments": {
    "city": "杭州"
  }
}
```

后端解析后执行工具。

---

## 8. 最小 Agent 循环

```ts
for (let step = 0; step < 3; step++) {
  const modelOutput = await callModel(messages);
  const action = parseModelAction(modelOutput);

  if (action.type === "final") {
    return action.answer;
  }

  if (action.type === "tool_call") {
    const toolResult = await runTool(action.toolName, action.arguments);

    messages.push({
      role: "tool",
      name: action.toolName,
      content: JSON.stringify(toolResult),
    });
  }
}

throw new Error("Too many tool calls");
```

为什么要限制 step？

因为模型可能陷入循环：

```text
调用工具 → 看结果 → 又调用工具 → 又看结果 ...
```

生产环境一定要限制最大工具调用次数。

---

## 9. 工具调用的安全边界

模型不能直接做这些事：

- 删除数据
- 修改权限
- 发邮件
- 下订单
- 支付
- 创建 API key
- 修改用户隐私数据

如果工具有副作用，必须让用户确认。

安全原则：

```text
读操作可以自动执行，但也要鉴权。
写操作必须谨慎，关键动作要用户确认。
```

例如：

```text
查询订单：可以自动
取消订单：必须用户确认
发送邮件：必须用户确认
删除文件：必须用户确认
```

---

## 10. 工具设计建议

### 10.1 工具要小而明确

不好：

```text
doAnything(input)
```

好：

```text
getOrderStatus(orderId)
searchDocs(query)
getWeather(city)
calculateTax(amount, region)
```

### 10.2 工具描述要清楚

模型靠 description 判断什么时候用工具。

不好：

```text
查询信息
```

好：

```text
根据订单 ID 查询订单的支付状态、发货状态和物流单号。
```

### 10.3 工具返回要简洁

不要把整个数据库对象都返回给模型。

只返回回答需要的信息。

---

## 11. Tool Calling 和 RAG 的关系

RAG 也可以做成一个工具：

```ts
const searchKnowledgeBaseTool = {
  name: "searchKnowledgeBase",
  description: "从项目学习文档中搜索和用户问题相关的资料片段",
  parameters: {
    type: "object",
    properties: {
      query: { type: "string" },
    },
    required: ["query"],
  },
};
```

这样模型可以决定：

```text
需要查文档 → 调 searchKnowledgeBase
需要查天气 → 调 getWeather
需要计算 → 调 calculate
不需要工具 → 直接回答
```

这就是工程 Agent 的雏形。

---

## 12. 跟学任务

- [ ] 设计一个 `getWeather` 工具 schema
- [ ] 写一个 `runGetWeather(args)` 函数
- [ ] 用 Zod 校验工具参数
- [ ] 写一个 `toolRegistry`
- [ ] 写一个 `runTool(name, args)`
- [ ] 用 JSON 协议模拟 tool calling
- [ ] 限制最大工具调用次数为 3
- [ ] 解释哪些工具必须让用户确认

---

## 下一步

```text
✅ 你已经知道 Tool Calling 怎么把模型和后端能力接起来
✅ 你已经知道模型不能绕过后端权限和校验

下一步 → 09_日志评估和部署.md
```
