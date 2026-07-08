// ═══════════════════════════════════════════════════════════════════════════════
// Local Embedding Engine — ONNX MiniLM-L6-v2 (384-dim)
//
// Runs entirely offline. No API calls. Works when OpenRouter is dead,
// when Ollama is dead, when credits are zero.
//
// Uses @xenova/transformers which auto-downloads + caches the ONNX model
// on first run (~23MB). Subsequent runs load from cache.
// ═══════════════════════════════════════════════════════════════════════════════

let pipeline: any = null;
let pipelinePromise: Promise<any> | null = null;

const MODEL_ID = "Xenova/all-MiniLM-L6-v2";

async function getEmbedder() {
  if (pipeline) return pipeline;
  if (pipelinePromise) return pipelinePromise;

  pipelinePromise = (async () => {
    // Dynamic import to avoid issues if not installed
    const { pipeline: createPipeline } = await import("@xenova/transformers");
    pipeline = await createPipeline("feature-extraction", MODEL_ID, {
      quantized: true, // use quantized model for speed on mobile/ARM
    });
    return pipeline;
  })();

  return pipelinePromise;
}

/** Embed a single text string → 384-dim float array */
export async function embed(text: string): Promise<Float32Array> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return output.data as Float32Array;
}

/** Embed multiple texts in batch → array of 384-dim vectors */
export async function embedBatch(texts: string[]): Promise<Float32Array[]> {
  if (texts.length === 0) return [];
  const embedder = await getEmbedder();
  const results: Float32Array[] = [];

  // Process in chunks to avoid memory issues on mobile
  const CHUNK = 32;
  for (let i = 0; i < texts.length; i += CHUNK) {
    const batch = texts.slice(i, i + CHUNK);
    for (const text of batch) {
      const output = await embedder(text, { pooling: "mean", normalize: true });
      results.push(output.data as Float32Array);
    }
  }

  return results;
}

/** Cosine similarity between two normalized vectors (already unit-norm from model) */
export function cosineSim(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
  }
  return dot; // vectors are already normalized, so dot = cosine
}

/** Preload the model (call at server startup so first request isn't slow) */
export async function warmup(): Promise<void> {
  console.log("[embeddings] Warming up local model...");
  const start = Date.now();
  await embed("warmup");
  console.log(`[embeddings] Model ready in ${Date.now() - start}ms`);
}
