# Node 应用基础

> 这部分是 Node 开发者学习大模型应用的工程底座。你不需要先成为算法工程师，但需要把 TypeScript、HTTP、流式响应、配置安全、异步错误、文件文本处理这些基础打牢。

---

## 为什么要加这一部分？

后面的 Hugging Face、RAG、Tool Calling、Agent 都会用到这些能力：

| 后续内容 | 需要的基础 |
|---|---|
| Node 集成 Hugging Face | TypeScript、环境变量、HTTP API |
| SSE 流式输出 | HTTP stream、ReadableStream、响应头 |
| Prompt 模板 | 字符串模板、类型设计、参数校验 |
| RAG 文档问答 | 文件读取、文本切块、JSON 数据结构 |
| Tool Calling | Schema、Zod 校验、异步函数、权限边界 |
| 日志评估部署 | requestId、错误处理、环境变量、Docker |

---

## 推荐学习顺序

```text
1. 01_TypeScript工程基础.md
   → 类型、接口、泛型、Zod，写可靠后端代码

2. 02_HTTP接口和SSE流式响应.md
   → 理解普通 API 和流式输出的差别

3. 03_环境变量和密钥安全.md
   → 学会保护 HF_TOKEN、API Key、配置

4. 04_异步编程和错误处理.md
   → 处理 Promise、超时、重试、统一错误

5. 05_文件处理和文本预处理.md
   → 为 RAG 做文档读取、清洗、切块
```

---

## 学完能做什么？

- [ ] 能写 TypeScript 类型和接口
- [ ] 能用 Zod 校验请求参数
- [ ] 能解释 SSE 为什么能边生成边显示
- [ ] 知道 token 和 API key 为什么不能放前端
- [ ] 能写统一错误处理中间件
- [ ] 能读取文本文件并切块
- [ ] 能为后面的 RAG 和 Tool Calling 打好工程基础
