# Node + Hugging Face Demo

一个完整的 Node/TypeScript 示例：用 Express 封装 Hugging Face 文本生成接口，并提供普通 JSON 返回和 SSE 流式返回。

## 运行

```bash
npm install
cp .env.example .env
npm run dev
```

然后打开：

```bash
http://localhost:3001
```

`.env` 里至少需要：

```bash
HF_TOKEN=hf_your_token_here
HF_MODEL=gpt2
PORT=3001
```

## 接口

### 健康检查

```bash
curl http://localhost:3001/health
```

### 普通生成

```bash
curl -X POST http://localhost:3001/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Deep learning is","maxNewTokens":80}'
```

### 流式生成

```bash
curl -N -X POST http://localhost:3001/api/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is deep learning?","maxNewTokens":80}'
```

## 生产环境建议

- 不要把 `HF_TOKEN` 放到前端。
- 给接口加用户鉴权和限流。
- 记录 prompt、模型、耗时、错误和 token 用量。
- 给模型调用加超时、重试和降级。
- 中文应用请换成中文能力更好的模型。
