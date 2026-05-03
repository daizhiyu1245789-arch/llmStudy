import { InferenceClient } from "@huggingface/inference";
import { assertHuggingFaceConfig, config } from "./config.js";

export type GenerateOptions = {
  prompt: string;
  maxNewTokens: number;
  temperature: number;
  topP: number;
};

assertHuggingFaceConfig();
const client = new InferenceClient(config.hfToken);

export async function generateText(options: GenerateOptions): Promise<string> {
  const output = await client.textGeneration({
    model: config.hfModel,
    inputs: options.prompt,
    parameters: {
      max_new_tokens: options.maxNewTokens,
      temperature: options.temperature,
      top_p: options.topP,
      return_full_text: false,
    },
  });

  return output.generated_text;
}

export async function* streamText(options: GenerateOptions): AsyncGenerator<string> {
  const stream = client.textGenerationStream({
    model: config.hfModel,
    inputs: options.prompt,
    parameters: {
      max_new_tokens: options.maxNewTokens,
      temperature: options.temperature,
      top_p: options.topP,
      return_full_text: false,
    },
  });

  for await (const chunk of stream) {
    const token = chunk.token?.text;
    if (token) {
      yield token;
    }
  }
}
