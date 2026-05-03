import "dotenv/config";

function optionalEnv(name: string): string | undefined {
  const value = process.env[name]?.trim();
  return value ? value : undefined;
}

function numberEnv(name: string, fallback: number): number {
  const raw = optionalEnv(name);
  if (!raw) return fallback;

  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new Error(`${name} must be a number`);
  }

  return value;
}

export const config = {
  port: numberEnv("PORT", 3001),
  corsOrigin: optionalEnv("CORS_ORIGIN") ?? "http://localhost:3001",
  hfToken: optionalEnv("HF_TOKEN"),
  hfModel: optionalEnv("HF_MODEL") ?? "gpt2",
};

export function assertHuggingFaceConfig(): void {
  if (!config.hfToken) {
    throw new Error("Missing HF_TOKEN. Create .env from .env.example and set your Hugging Face token.");
  }
}
