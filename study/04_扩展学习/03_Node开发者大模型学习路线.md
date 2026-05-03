# Node 开发者大模型学习路线

> 你是 Node 开发者，不需要一开始就把自己训练成算法工程师。更好的路线是：先懂原理，再把模型能力接进真实应用。

---

## 1. 你现在的优势

Node 开发者学大模型有几个天然优势：

- 熟悉 HTTP API、鉴权、限流、缓存、日志
- 熟悉异步、流式响应、SSE、WebSocket
- 熟悉前后端联调和产品交互
- 熟悉 npm 生态，能很快集成 SDK
- 更容易把大模型做成真实工具，而不是只停留在 notebook

大模型应用工程里，很多关键能力其实不是“训练模型”，而是：

```
模型调用
Prompt 组织
上下文管理
流式输出
文档检索 RAG
工具调用
任务编排
日志与评估
成本控制
部署监控
```

这些都很适合 Node / TypeScript 开发者切入。

---

## 2. 建议你补的知识模块

### 2.1 TypeScript 后端基础

如果你还没系统用 TS 写 Node 服务，建议补：

- 类型建模
- 环境变量管理
- 请求参数校验
- 错误处理中间件
- 日志
- 流式响应

大模型接口返回内容不稳定，类型和校验会很救命。

### 2.2 SSE 流式输出

聊天产品里，流式输出几乎是标配。

```
用户发问题
  ↓
后端请求模型
  ↓
模型一个 token 一个 token 返回
  ↓
后端用 SSE 推给浏览器
  ↓
前端边收边渲染
```

你要重点理解：

- `Content-Type: text/event-stream`
- `data: ...\n\n`
- 连接关闭和错误处理
- 前端如何用 `fetch` 读取 stream

### 2.3 Prompt 工程

Prompt 不是玄学，更多是输入协议设计。

你要会写：

```
system：定义角色、边界、输出格式
user：用户真实问题
context：检索到的资料
schema：要求模型按 JSON 输出
examples：给少量示例
```

### 2.4 RAG 文档问答

RAG 是大模型应用最常见的落地方向。

```
文档切块
  ↓
生成 embedding
  ↓
存向量数据库
  ↓
用户问题生成 embedding
  ↓
相似度检索
  ↓
把相关文档塞进 prompt
  ↓
模型回答并引用来源
```

Node 侧需要掌握：

- 文件上传和解析
- 文本切块
- embedding 调用
- 向量库接入
- 检索结果拼 prompt
- 引用来源展示

### 2.5 工具调用 Tool Calling

让模型不只是“说”，还能“做”。

例子：

```
用户：查一下今天订单 123 的物流
模型：决定调用 getOrderShipping(orderId)
后端：执行真实函数
模型：根据函数结果回答用户
```

你要会设计：

- 工具 schema
- 参数校验
- 权限控制
- 执行超时
- 工具结果回填

### 2.6 评估和日志

大模型应用上线后，最怕“看起来能用，但不知道什么时候坏”。

建议从一开始就记录：

- prompt
- model
- latency
- token 使用量
- 错误信息
- 用户反馈
- 检索命中文档
- 最终回答

---

## 3. Node 开发者推荐学习顺序

```
1. 跑通本项目 tinyllm，理解模型训练和生成的底层流程
2. 学 Hugging Face 基础，知道模型、Tokenizer、pipeline 是什么
3. 用 Node 接 Hugging Face API，做普通生成接口
4. 加 SSE，做流式输出
5. 加 Prompt 模板，固定输出格式
6. 加 embedding 和向量检索，做 RAG
7. 加工具调用，做一个小 Agent
8. 加日志、缓存、限流、评估、部署
```

---

## 4. Node 里集成 Hugging Face 的两条路

### 路线 A：服务端调用 Hugging Face Inference

适合：

- 后端服务
- 需要隐藏 token
- 需要接入数据库、用户系统、日志
- 需要做流式输出

常用包：

```bash
npm install @huggingface/inference
```

官方文档：

- https://huggingface.co/docs/huggingface.js/inference/modules
- https://huggingface.co/docs/text-generation-inference/en/conceptual/streaming

本项目示例代码：

```
examples/node-huggingface/
```

### 路线 B：Transformers.js 本地/浏览器推理

适合：

- 小模型
- 浏览器端离线推理
- Electron
- 不想把用户数据发到服务端

常用包：

```bash
npm install @huggingface/transformers
```

官方文档：

- https://huggingface.co/docs/transformers.js

注意：浏览器或本地推理更吃设备性能，模型也通常需要 ONNX / quantized 版本。

---

## 5. 本项目已经补齐的实战章节

现在可以按这个顺序继续学：

```
03_项目实战/04_Node集成HuggingFace.md
03_项目实战/05_Prompt模板和结构化输出.md
03_项目实战/06_Embedding和向量检索.md
03_项目实战/07_RAG文档问答.md
03_项目实战/08_ToolCalling工具调用.md
03_项目实战/09_日志评估和部署.md
```

如果你想做成完整作品集，最推荐做：

```
Node + React + Hugging Face + RAG 文档问答
```

这个方向非常贴近真实业务。

---

## 6. 学习检查表

- [ ] 知道 Hugging Face Hub、Inference、Transformers.js 的区别
- [ ] 能用 Node 调一个模型生成文本
- [ ] 能用 SSE 把 token 流式返回给前端
- [ ] 知道 prompt 应该当成接口协议设计
- [ ] 知道 RAG 的完整链路
- [ ] 知道工具调用和工程 Agent 的关系
- [ ] 知道上线后要记录哪些日志和指标

---

## 下一步

```
✅ 你已经有了 Node 开发者的大模型学习路线
✅ 知道该优先补服务端集成、流式输出、RAG、工具调用

下一步 → 03_项目实战/04_Node集成HuggingFace.md
```
