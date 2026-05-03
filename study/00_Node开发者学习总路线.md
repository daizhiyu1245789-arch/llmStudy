# Node 开发者大模型学习总路线

> 目标：跟着这个目录学一遍，最后能用 Node 做出一个能跑、能流式输出、能接 Hugging Face、能继续扩展 RAG 和工具调用的 AI 应用。

---

## 0. 你最终要学会什么？

你不需要一开始就成为算法工程师。更适合 Node 开发者的目标是：

```text
懂一点模型底层
  ↓
会调用真实大模型
  ↓
会封装 Node API
  ↓
会做流式输出
  ↓
会做 RAG 文档问答
  ↓
会做工具调用和工程 Agent
  ↓
会部署、记录日志、控制成本
```

最后你应该能独立做一个：

```text
个人知识库问答助手
```

功能包括：

- 普通聊天
- 流式输出
- 接 Hugging Face 模型
- 上传文档
- 文档切块和检索
- 根据资料回答
- 调用后端工具
- 记录请求日志

---

## 1. 为什么 Node 开发者要懂一点底层？

只会调 API 也能做 demo，但一到真实项目就容易卡住：

- 为什么模型有时胡说？
- 为什么 prompt 一长效果变差？
- temperature 到底怎么调？
- 为什么中文模型和英文模型差别很大？
- 为什么 RAG 检索到了资料，模型还是答错？
- 为什么流式输出不是一次性返回？
- 为什么同一个问题每次答案不一样？

你需要掌握的底层程度是：

```text
Tokenizer：文字怎么变成 token
Embedding：token 怎么变成向量
Attention：模型怎么关注上下文
GPT：为什么是预测下一个 token
Loss：训练时怎么衡量错多少
Sampling：生成时怎么选下一个 token
RAG：为什么要把外部资料塞给模型
Agent：模型怎么选择调用工具
```

够了。先不用死磕所有公式。

---

## 2. 六阶段学习路线

| 阶段 | 学什么 | 你会得到什么 |
|---|---|---|
| 1 | Node 应用基础 | 打好 TypeScript、SSE、配置安全、异步错误、文件处理底座 |
| 2 | Python / PyTorch / tinyllm | 跑通小模型训练和生成 |
| 3 | 数学直觉 / 机器学习 / 深度学习 | 建立算法和训练地图 |
| 4 | Tokenizer / Attention / GPT | 看懂大模型核心结构 |
| 5 | Node 集成 Hugging Face / Prompt / RAG / Tool Calling | 做真实 AI 应用 |
| 6 | 日志 / 评估 / 部署 | 把 demo 变成可维护服务 |

---

## 3. 推荐学习顺序

### 第一阶段：先补 Node 应用基础

这些是后面做 Hugging Face、RAG、Tool Calling 的工程底座：

```text
1. study/01_Node应用基础/README.md
2. study/01_Node应用基础/01_TypeScript工程基础.md
3. study/01_Node应用基础/02_HTTP接口和SSE流式响应.md
4. study/01_Node应用基础/03_环境变量和密钥安全.md
5. study/01_Node应用基础/04_异步编程和错误处理.md
6. study/01_Node应用基础/05_文件处理和文本预处理.md
7. study/01_Node应用基础/06_Node大模型应用项目结构.md
```

目标：

- 会用 TypeScript 和 Zod 固定接口边界
- 知道 SSE 怎么实现流式输出
- 知道 API Key 为什么不能放前端
- 会处理异步、超时、错误和 requestId
- 会做文件读取、文本清洗和切块
- 知道大模型项目怎么分层

### 第二阶段：先跑起来

先别急着看完所有理论。你是工程师，先跑通会更有感觉。

```text
8. study/01_Python基础/01_Python_vs_JS语法对比.md
9. study/01_Python基础/03_PyTorch快速入门.md
10. study/01_Python基础/04_动手跑第一个模型.md
11. study/03_项目实战/01_把tinyllm跑起来.md
```

目标：

- 装好 Python 环境
- 跑训练命令
- 看到 loss 下降
- 用 checkpoint 生成文本

### 第三阶段：补底层概念

```text
12. study/02_深度学习核心/00_大模型数学直觉.md
13. study/02_深度学习核心/01_机器学习是什么.md
14. study/02_深度学习核心/02_神经网络基础.md
15. study/02_深度学习核心/06_神经网络进阶.md
16. study/02_深度学习核心/08_深度学习概览.md
17. study/02_深度学习核心/07_常用机器学习算法.md
18. study/02_深度学习核心/09_强化学习入门.md
```

目标：

- 解释训练、loss、梯度、优化器
- 解释向量、矩阵、softmax、cosine similarity
- 区分机器学习、深度学习、大模型
- 知道常用算法适合什么任务
- 知道强化学习和 RLHF 是什么

### 第四阶段：理解大模型核心

```text
19. study/02_深度学习核心/04_Tokenizer分词器.md
20. study/02_深度学习核心/03_Attention机制.md
21. study/02_深度学习核心/05_大模型入门.md
```

目标：

- 知道文本为什么要变成 token
- 知道 GPT 为什么只能看前文
- 知道 Attention 和 MLP 分别做什么
- 能把 tinyllm 和真实大模型对应起来

### 第五阶段：开始 Node 大模型应用

```text
22. study/04_扩展学习/03_Node开发者大模型学习路线.md
23. study/03_项目实战/04_Node集成HuggingFace.md
24. study/03_项目实战/05_Prompt模板和结构化输出.md
25. study/03_项目实战/06_Embedding和向量检索.md
26. study/03_项目实战/07_RAG文档问答.md
27. study/03_项目实战/08_ToolCalling工具调用.md
28. study/03_项目实战/09_日志评估和部署.md
```

目标：

- 用 Node 调 Hugging Face
- 封装 /api/generate
- 封装 /api/generate/stream
- 用浏览器页面联调
- 知道 token 为什么不能放前端
- 设计 Prompt 模板和结构化输出
- 做 RAG 文档问答
- 让模型调用后端工具
- 加日志、评估、限流和部署思路

### 第五阶段：进入真实业务能力

这一阶段已经拆到了项目实战文档里：

```text
Prompt 模板和结构化输出
Embedding 和向量检索
RAG 文档问答
Tool Calling 工具调用
日志、评估和部署
```

目标：

- 做文档问答
- 做知识库客服
- 做代码解释工具
- 做能调用业务接口的小助手

### 第六阶段：上线和维护

这部分先读：

```text
study/03_项目实战/09_日志评估和部署.md
```

然后继续补：

```text
日志平台接入
数据库持久化
用户系统和权限
Docker / 云部署
线上评估和监控
```

目标：

- 知道线上服务怎么排错
- 知道怎么控制模型调用成本
- 知道怎么判断一次改动有没有让回答变差

---

## 4. 每阶段小作业

### 作业 1：训练 tinyllm

- [ ] 跑一次训练
- [ ] 看懂 loss 日志
- [ ] 用 checkpoint 生成文本
- [ ] 调一次 temperature 对比输出

### 作业 2：解释模型流程

能画出并解释：

```text
文本 → token ID → embedding → Transformer → logits → 采样 → 文本
```

### 作业 3：Node 接模型

能实现：

```text
POST /api/generate
输入：prompt
输出：generatedText
```

### 作业 4：流式输出

能实现：

```text
POST /api/generate/stream
返回：SSE token 流
```

### 作业 5：RAG

能讲清：

```text
文档 → 切块 → embedding → 向量库 → 检索 → prompt → 回答
```

---

## 5. 判断自己是否真的会了

- [ ] 能换一个 Hugging Face 模型
- [ ] 能把普通返回改成流式返回
- [ ] 能给接口加参数校验
- [ ] 能处理模型调用失败
- [ ] 能让前端边收边显示文本
- [ ] 能把 prompt 抽成模板
- [ ] 能把文档内容塞进 prompt
- [ ] 能给回答附上引用来源
- [ ] 能记录每次请求的耗时和错误
- [ ] 能写 README 让别人跑起来

做到这些，你就不是“看过大模型”，而是“能做大模型应用”。
