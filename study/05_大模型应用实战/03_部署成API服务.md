# 部署成 API 服务

> 把训练好的模型封装成 HTTP 接口，前端页面或后端 Node.js 都能调用。

---

## 1. 整体架构

```
前端页面 / Node.js 后端
        ↓ HTTP POST
FastAPI 服务（Python）
        ↓
加载 checkpoint
        ↓
tinyllm GPT 模型
        ↓
返回生成的文本
```

类似你写一个 Express 接口：
```js
// POST /generate
app.post('/generate', (req, res) => {
  const { prompt } = req.body;
  const result = callPythonModel(prompt);
  res.json({ text: result });
});
```

---

## 2. 安装 FastAPI 和 uvicorn

```bash
pip install fastapi uvicorn
```

- **FastAPI**：Python 的现代 Web 框架（类似 Express）
- **uvicorn**：ASGI 服务器（类似 Node.js 的 http 模块）

---

## 3. 创建 API 服务文件

新建文件 `server.py`：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np

from tinyllm.model import GPT, GPTConfig
from tinyllm.checkpoint import load_checkpoint
from tinyllm.tokenizer import tokenizer_from_extra

# ===== 创建 FastAPI 应用 =====
app = FastAPI(title="tinyllm 文本生成 API")

# ===== CORS 配置（允许跨域，前端才能调）=====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 允许所有来源（生产环境改成具体域名）
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 模型加载（启动时一次性加载，不重复加载）=====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = load_checkpoint("checkpoints/latest.pt", map_location=DEVICE)
tokenizer = tokenizer_from_extra(ckpt.get("extra"))
model = GPT(GPTConfig(**ckpt["model_cfg"]))
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()


# ===== API 接口 =====
@app.post("/generate")
def generate(prompt: str, max_tokens: int = 100, temperature: float = 0.9, top_k: int = 50):
    """
    文本生成接口。

    参数：
        prompt: 生成的开头文本
        max_tokens: 最大生成 token 数
        temperature: 随机性（越大越有创意）
        top_k: 只从概率最高的 k 个 token 中选择
    """
    # 编码
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # 生成
    out = model.generate(
        idx,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k if top_k > 0 else None,
    )

    # 解码
    text = tokenizer.decode(out[0].tolist())
    return {"text": text}


@app.get("/health")
def health():
    """健康检查接口"""
    return {"status": "ok", "device": str(DEVICE)}


# ===== 启动服务 =====
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
```

---

## 4. 启动服务

```bash
python server.py
```

或者用 uvicorn 直接运行：

```bash
uvicorn server:app --host 0.0.0.0 --port 3000 --reload
```

`--reload` = 代码改了就自动重启（开发模式用）。

---

## 5. 测试 API

### 5.1 用 curl 测试

```bash
curl -X POST "http://localhost:3000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "今天", "max_tokens": 50, "temperature": 0.9}'
```

预期返回：
```json
{"text": "今天天气真是不错，阳光温暖地洒在窗台上..."}
```

健康检查：
```bash
curl http://localhost:3000/health
```

### 5.2 用浏览器测试

打开浏览器访问：

```
http://localhost:3000/docs
```

FastAPI 自动生成文档页面（Swagger UI），可以直接在线测试接口。

---

## 6. 前端调用（JavaScript / Node.js）

### 浏览器端调用

```html
<!DOCTYPE html>
<html>
<head>
  <title>GPT 生成</title>
</head>
<body>
  <input id="prompt" placeholder="输入提示词" />
  <button onclick="generate()">生成</button>
  <p id="output"></p>

  <script>
    async function generate() {
      const prompt = document.getElementById("prompt").value;
      const response = await fetch("http://localhost:3000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          max_tokens: 50,
          temperature: 0.9,
          top_k: 50
        })
      });
      const data = await response.json();
      document.getElementById("output").innerText = data.text;
    }
  </script>
</body>
</html>
```

### Node.js 后端调用

```js
// Node.js 调用 Python API
async function generate(prompt) {
  const response = await fetch("http://localhost:3000/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, max_tokens: 50, temperature: 0.9 })
  });
  const data = await response.json();
  return data.text;
}

generate("今天").then(console.log);
// → "今天天气真是不错..."
```

---

## 7. 用 Node.js 直接调 Python 脚本

如果不想单独起服务，可以用 `child_process` 直接调：

```js
const { spawn } = require('child_process');

function generate(prompt, callback) {
  const py = spawn('python', [
    '-c',
    `
import sys
sys.path.insert(0, '.')
from tinyllm.checkpoint import load_checkpoint
from tinyllm.model import GPT, GPTConfig
from tinyllm.tokenizer import tokenizer_from_extra
import torch

ckpt = load_checkpoint("checkpoints/latest.pt", map_location="cpu")
tokenizer = tokenizer_from_extra(ckpt.get("extra"))
model = GPT(GPTConfig(**ckpt["model_cfg"]))
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()

ids = tokenizer.encode("${prompt}")
if not ids:
    ids = [0]
idx = torch.tensor([ids], dtype=torch.long)
out = model.generate(idx, max_new_tokens=50, temperature=0.9)
print(tokenizer.decode(out[0].tolist()))
`
  ]);

  let result = '';
  py.stdout.on('data', data => { result += data.toString(); });
  py.stdout.on('end', () => callback(result.trim()));
}

generate('今天', console.log);
```

---

## 8. 部署到云服务器

### 8.1 把项目传到服务器

```bash
# 用 scp 传文件
scp -r e:/python/llmStudy user@your-server:/home/user/
```

### 8.2 在服务器上安装依赖

```bash
ssh user@your-server
cd tinyllm
python -m venv .venv
source .venv/bin/activate
pip install torch fastapi uvicorn
```

### 8.3 用 systemd 后台运行

```bash
# /etc/systemd/system/tinyllm.service
[Unit]
Description=tinyllm API Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/tinyllm
ExecStart=/home/ubuntu/tinyllm/.venv/bin/python -m uvicorn server:app --host 0.0.0.0 --port 3000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable tinyllm
sudo systemctl start tinyllm
sudo systemctl status tinyllm  # 查看状态
```

---

## 9. Docker 部署（可选）

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "3000"]
```

```bash
# 构建镜像
docker build -t tinyllm-api .

# 运行容器
docker run -d -p 3000:3000 --name tinyllm tinyllm-api

# 访问
curl http://localhost:3000/health
```

---

## 10. 你现在做到的事情

```
✅ 创建了 FastAPI 服务
✅ 配置了 CORS 支持跨域
✅ 实现了 /generate 接口
✅ 用 curl 测试了接口
✅ 用浏览器页面调用了 API
✅ 学会了 Node.js 调用 Python API
✅ 学会了部署到云服务器
```

---

## 部署方式对比

| 方式 | 优点 | 缺点 |
|------|------|------|
| 本地 API（开发） | 简单，调试方便 | 只能本地访问 |
| 云服务器 + systemd | 稳定，后台常驻 | 需要服务器 |
| Docker | 跨平台，一键部署 | 需要 Docker 环境 |
| 云函数 | 按调用计费，不用管服务器 | 冷启动慢，首次加载慢 |
| Hugging Face Inference API | 免费，不用自己部署 | 调用量限制，模型公开 |
