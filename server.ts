import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import { runSemanticPipeline, serializeGrid } from "./semantic/pipeline.ts";
import { warmup } from "./semantic/embeddings.ts";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

const SCL_SYSTEM = `You are a semantic compression engine. Respond ONLY in Semantic Compression Language (SCL).

Format:
@concept_name [
  key → value
  subject → relation → object
  uncertainty → confidence
]

Rules:
- Use @<concept_name> [ ... ] blocks. Use lowercase_snake_case for concept names.
- Each line inside is: key → value (or subject → relation → object).
- Include exactly one "uncertainty → <0.0-1.0>" line per concept.
- Use lowercase_snake_case for all keys.
- No prose, no markdown, no explanation outside SCL blocks.
- Compress the user's request into 2-5 concepts.
- Sort properties alphabetically within each concept.
- Use canonical short forms: no articles, no filler words in values.`;

// Fetch available free models from OpenRouter
app.get("/api/models", async (_req, res) => {
  try {
    const resp = await fetch("https://openrouter.ai/api/v1/models", {
      headers: { Authorization: `Bearer ${OPENROUTER_API_KEY}` },
    });
    const data = await resp.json() as any;
    const freeModels = (data.data || [])
      .filter((m: any) => m.id?.includes(":free") || m.pricing?.prompt === "0")
      .map((m: any) => ({ id: m.id, name: m.name }))
      .slice(0, 24); // cap at 24 for the grid
    res.json(freeModels);
  } catch (e: any) {
    res.status(500).json({ error: e.message });
  }
});

// SSE endpoint: streams SCL from all free models in parallel
app.get("/api/stream", async (req, res) => {
  const prompt = req.query.prompt as string;
  if (!prompt) {
    res.status(400).json({ error: "prompt required" });
    return;
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  // First fetch models
  let models: { id: string; name: string }[] = [];
  try {
    const resp = await fetch("https://openrouter.ai/api/v1/models", {
      headers: { Authorization: `Bearer ${OPENROUTER_API_KEY}` },
    });
    const data = await resp.json() as any;
    models = (data.data || [])
      .filter((m: any) => m.id?.includes(":free") || m.pricing?.prompt === "0")
      .map((m: any) => ({ id: m.id, name: m.name }))
      .slice(0, 24);
  } catch (e: any) {
    res.write(`data: ${JSON.stringify({ type: "error", error: e.message })}\n\n`);
    res.end();
    return;
  }

  // Filter out non-chat models (music/audio models)
  models = models.filter((m) => !m.id.includes("lyria"));

  // Send model list
  res.write(`data: ${JSON.stringify({ type: "models", models })}\n\n`);

  const controller = new AbortController();
  req.on("close", () => controller.abort());

  const MAX_RETRIES = 3;
  const STAGGER_MS = 500; // stagger requests to avoid simultaneous rate-limit hits

  async function streamOneModel(model: { id: string; name: string }, idx: number, attempt: number): Promise<void> {
    if (controller.signal.aborted) return;

    try {
      const resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${OPENROUTER_API_KEY}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: model.id,
          stream: true,
          messages: [
            { role: "system", content: SCL_SYSTEM },
            { role: "user", content: prompt },
          ],
          max_tokens: 400,
        }),
        signal: controller.signal,
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({})) as any;
        const retryAfter = errData.error?.metadata?.retry_after_seconds;

        // Retry on 429 with backoff
        if (resp.status === 429 && attempt < MAX_RETRIES && retryAfter) {
          const waitMs = (retryAfter + 1) * 1000;
          res.write(`data: ${JSON.stringify({ type: "model_retry", idx, modelId: model.id, attempt: attempt + 1, waitMs })}\n\n`);
          await new Promise((r) => setTimeout(r, waitMs));
          return streamOneModel(model, idx, attempt + 1);
        }

        res.write(`data: ${JSON.stringify({ type: "model_error", idx, modelId: model.id, error: errData.error?.message || `HTTP ${resp.status}` })}\n\n`);
        return;
      }

      const reader = resp.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let pending = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        pending += decoder.decode(value, { stream: true });
        const lines = pending.split("\n");
        pending = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith("data: ")) continue;
          const payload = trimmed.slice(6);
          if (payload === "[DONE]") break;

          try {
            const json = JSON.parse(payload);
            const delta = json.choices?.[0]?.delta?.content;
            if (delta) {
              res.write(`data: ${JSON.stringify({ type: "token", idx, modelId: model.id, token: delta })}\n\n`);
            }
          } catch { /* skip */ }
        }
      }

      res.write(`data: ${JSON.stringify({ type: "model_done", idx, modelId: model.id })}\n\n`);
    } catch (e: any) {
      if (e?.name === "AbortError") return;

      // Retry on network errors
      if (attempt < MAX_RETRIES) {
        const waitMs = (attempt + 1) * 2000;
        res.write(`data: ${JSON.stringify({ type: "model_retry", idx, modelId: model.id, attempt: attempt + 1, waitMs })}\n\n`);
        await new Promise((r) => setTimeout(r, waitMs));
        return streamOneModel(model, idx, attempt + 1);
      }

      res.write(`data: ${JSON.stringify({ type: "model_error", idx, modelId: model.id, error: String(e) })}\n\n`);
    }
  }

  // Stagger model requests in batches of 4 to avoid rate-limit storms
  const BATCH_SIZE = 4;
  const allPromises: Promise<void>[] = [];

  for (let i = 0; i < models.length; i += BATCH_SIZE) {
    if (controller.signal.aborted) break;
    const batch = models.slice(i, i + BATCH_SIZE);
    for (let j = 0; j < batch.length; j++) {
      allPromises.push(streamOneModel(batch[j], i + j, 0));
    }
    // Stagger before launching next batch
    if (i + BATCH_SIZE < models.length) {
      await new Promise((r) => setTimeout(r, STAGGER_MS));
    }
  }

  await Promise.allSettled(allPromises);
  res.write(`data: ${JSON.stringify({ type: "done" })}\n\n`);
  res.end();
});

// ── Semantic pipeline endpoint ────────────────────────────────────────────────
// POST /api/semantic — takes model buffers, returns semantic grid
app.post("/api/semantic", async (req, res) => {
  const { models } = req.body as { models: { modelIdx: number; buffer: string; error?: string }[] };
  if (!models || !Array.isArray(models)) {
    res.status(400).json({ error: "models array required" });
    return;
  }

  try {
    const grid = await runSemanticPipeline(models);
    res.json(serializeGrid(grid));
  } catch (e: any) {
    console.error("[semantic] Pipeline error:", e);
    res.status(500).json({ error: e.message });
  }
});

const PORT = 3001;
app.listen(PORT, async () => {
  console.log(`SCL Pixel Consensus backend on http://localhost:${PORT}`);
  // Warmup embedding model in background
  warmup().catch((e) => console.error("[embeddings] Warmup failed:", e));
});
