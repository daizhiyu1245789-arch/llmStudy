# HTTP 接口和 SSE 流式响应

> 大模型聊天为什么能“一个字一个字冒出来”？核心就是流式响应。Node 开发者必须搞懂普通 HTTP 和 SSE 的区别。

---

## 0. 学完这一篇你会什么？

你会学会：

- 普通 HTTP API 的请求响应模型
- 为什么大模型需要流式输出
- SSE 的响应头和数据格式
- Node/Express 怎么写 SSE
- 前端怎么读取 stream
- SSE、WebSocket、普通 API 怎么选

---

## 1. 普通 HTTP 接口

普通接口是一次请求，一次完整响应。

```text
浏览器发请求
  ↓
Node 处理
  ↓
Node 等模型完整生成
  ↓
一次性返回 JSON
```

Express 示例：

```ts
app.post("/api/generate", async (req, res) => {
  const text = await generateText(req.body.prompt);

  res.json({
    generatedText: text,
  });
});
```

优点：

- 简单
- 好调试
- 适合短任务

缺点：

- 模型生成慢时，用户一直等
- 长回答体验不好

---

## 2. 为什么大模型需要流式输出？

模型生成文本通常是一个 token 一个 token 生成：

```text
Deep
Deep learning
Deep learning is
Deep learning is a
...
```

如果等全部生成完再返回，用户会觉得卡。

流式输出可以：

- 更快看到结果
- 减少等待焦虑
- 支持中途停止
- 更像 ChatGPT 的体验

---

## 3. SSE 是什么？

SSE = Server-Sent Events。

它允许服务端持续向浏览器推送文本事件。

响应头：

```ts
res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
res.setHeader("Cache-Control", "no-cache, no-transform");
res.setHeader("Connection", "keep-alive");
```

数据格式：

```text
data: {"token":"Hello"}

data: {"token":" world"}

data: {"done":true}

```

注意：每条事件后面都有一个空行，也就是 `\n\n`。

---

## 4. Express 里写 SSE

```ts
app.post("/api/generate/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  for await (const token of streamText(req.body.prompt)) {
    res.write(`data: ${JSON.stringify({ token })}\n\n`);
  }

  res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
  res.end();
});
```

关键点：

- `flushHeaders()` 先把响应头发出去
- `res.write()` 可以多次写入
- 每个 SSE event 用 `data: ...\n\n`
- 最后一定 `res.end()`

---

## 5. 前端读取 stream

用 `fetch` 读取响应体：

```js
const response = await fetch("/api/generate/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt }),
});

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
    const line = event
      .split("\n")
      .find((item) => item.startsWith("data: "));

    if (!line) continue;

    const data = JSON.parse(line.slice(6));
    if (data.token) {
      output.textContent += data.token;
    }
  }
}
```

---

## 6. SSE 和 WebSocket 怎么选？

| 方式 | 特点 | 适合 |
|---|---|---|
| 普通 HTTP | 一次请求一次响应 | 短任务、普通接口 |
| SSE | 服务端持续推送 | 大模型流式输出 |
| WebSocket | 双向长连接 | 实时协作、游戏、复杂会话 |

大模型聊天入门：

```text
优先 SSE。
```

因为它简单，浏览器支持好，和 HTTP 部署模型接近。

---

## 7. 流式错误怎么处理？

如果响应头还没发，可以正常返回 JSON 错误。

如果已经开始流式输出，就只能继续写 SSE 错误事件：

```ts
if (res.headersSent) {
  res.write(`data: ${JSON.stringify({ error: "模型调用失败" })}\n\n`);
  res.end();
  return;
}
```

前端收到：

```js
if (data.error) {
  throw new Error(data.error);
}
```

---

## 8. 跟学任务

- [ ] 写一个普通 `/api/generate`
- [ ] 写一个 `/api/generate/stream`
- [ ] 用 `res.write()` 连续输出 5 段文本
- [ ] 前端用 `fetch + reader` 读取 stream
- [ ] 故意让后端中途报错，观察前端如何处理
- [ ] 解释 SSE 和 WebSocket 的区别

---

## 下一步

```text
✅ 你已经理解为什么大模型聊天需要流式输出
✅ 你已经知道 SSE 在 Node 和前端两侧怎么写

下一步 → 03_环境变量和密钥安全.md
```
