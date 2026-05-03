import cors from "cors";
import express, { type NextFunction, type Request, type Response } from "express";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { z } from "zod";
import { config } from "./config.js";
import { generateText, streamText } from "./huggingface.js";

const app = express();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const publicDir = path.resolve(__dirname, "../public");

const generateSchema = z.object({
  prompt: z.string().trim().min(1, "prompt is required").max(8000),
  maxNewTokens: z.coerce.number().int().min(1).max(512).default(120),
  temperature: z.coerce.number().min(0.1).max(2).default(0.8),
  topP: z.coerce.number().min(0.1).max(1).default(0.95),
});

app.use(cors({ origin: config.corsOrigin }));
app.use(express.json({ limit: "1mb" }));
app.use(express.static(publicDir));

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    model: config.hfModel,
  });
});

app.post("/api/generate", async (req, res, next) => {
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

app.post("/api/generate/stream", async (req, res, next) => {
  try {
    const input = generateSchema.parse(req.body);

    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.flushHeaders();

    for await (const token of streamText(input)) {
      res.write(`data: ${JSON.stringify({ token })}\n\n`);
    }

    res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
    res.end();
  } catch (error) {
    if (res.headersSent) {
      res.write(`data: ${JSON.stringify({ error: getErrorMessage(error) })}\n\n`);
      res.end();
      return;
    }

    next(error);
  }
});

app.use((error: unknown, _req: Request, res: Response, _next: NextFunction) => {
  if (error instanceof z.ZodError) {
    res.status(400).json({
      error: "Invalid request",
      issues: error.issues,
    });
    return;
  }

  res.status(500).json({
    error: getErrorMessage(error),
  });
});

app.listen(config.port, () => {
  console.log(`Node + Hugging Face demo running at http://localhost:${config.port}`);
});

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  return "Unknown error";
}
