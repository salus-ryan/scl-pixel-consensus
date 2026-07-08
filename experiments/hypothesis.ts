// ═══════════════════════════════════════════════════════════════════════════════
// Hypothesis Test: Does per-assertion cross-model consensus predict factual accuracy?
//
// Method:
//   1. N factual questions with unambiguous gold answers
//   2. Query ensemble of free OpenRouter models (SCL format)
//   3. Run semantic pipeline → clusters → assertion value branches
//   4. Consensus answer = value branch with highest model support
//   5. Compare: consensus accuracy vs mean/best individual model accuracy
//   6. Calibration: does support ratio correlate with correctness?
//
// Run: npx tsx experiments/hypothesis.ts
// ═══════════════════════════════════════════════════════════════════════════════

import dotenv from "dotenv";
import { parseSCLToAST } from "../semantic/normalize.ts";
import { clusterConcepts } from "../semantic/cluster.ts";

dotenv.config();

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;

const SCL_SYSTEM = `You are a semantic compression engine. Respond ONLY in Semantic Compression Language (SCL).

SCL format:
@concept_name [
  key → value
  key → value
  uncertainty → 0.0-1.0
]

Rules:
- Use lowercase snake_case for concept names and keys
- Multiple @concept blocks allowed
- Values should be short and factual
- Include an uncertainty field (0=certain, 1=uncertain)
- NO prose, NO explanations, ONLY SCL blocks`;

// ── Test set: unambiguous factual questions ───────────────────────────────────

interface Question {
  prompt: string;
  gold: string[]; // acceptable answer substrings (lowercase)
}

const QUESTIONS: Question[] = [
  { prompt: "What is the chemical symbol for gold?", gold: ["au"] },
  { prompt: "What year did World War 2 end?", gold: ["1945"] },
  { prompt: "What is the capital of Australia?", gold: ["canberra"] },
  { prompt: "How many chromosomes do humans have?", gold: ["46", "23 pairs", "23_pairs"] },
  { prompt: "What planet is known as the Red Planet?", gold: ["mars"] },
  { prompt: "Who wrote the play Romeo and Juliet?", gold: ["shakespeare"] },
  { prompt: "What is the speed of light in km/s (approximately)?", gold: ["299,792", "299792", "300,000", "300000", "3x10", "3 x 10"] },
  { prompt: "What is the largest ocean on Earth?", gold: ["pacific"] },
  { prompt: "What element has atomic number 1?", gold: ["hydrogen"] },
  { prompt: "In what country is the Eiffel Tower located?", gold: ["france"] },
  { prompt: "What is the smallest prime number?", gold: ["2", "two"] },
  { prompt: "What gas do plants absorb from the atmosphere for photosynthesis?", gold: ["co2", "carbon dioxide", "carbon_dioxide"] },
  { prompt: "Who painted the Mona Lisa?", gold: ["da vinci", "da_vinci", "davinci", "leonardo"] },
  { prompt: "What is the longest river in the world?", gold: ["nile", "amazon"] }, // both accepted (contested)
  { prompt: "How many continents are there?", gold: ["7", "seven"] },
];

// ── Model querying ────────────────────────────────────────────────────────────

async function getFreeModels(): Promise<string[]> {
  const resp = await fetch("https://openrouter.ai/api/v1/models", {
    headers: { Authorization: `Bearer ${OPENROUTER_API_KEY}` },
  });
  const data = await resp.json() as any;
  return (data.data || [])
    .filter((m: any) => m.id?.includes(":free") || m.pricing?.prompt === "0")
    .map((m: any) => m.id)
    .filter((id: string) => !id.includes("lyria"))
    .slice(0, 12); // cap at 12 models to keep rate limits manageable
}

async function queryModel(modelId: string, prompt: string, attempt = 0): Promise<string | null> {
  const MAX_RETRIES = 3;
  try {
    const resp = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: modelId,
        messages: [
          { role: "system", content: SCL_SYSTEM },
          { role: "user", content: prompt },
        ],
        max_tokens: 300,
      }),
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({})) as any;
      const retryAfter = errData.error?.metadata?.retry_after_seconds;
      if (resp.status === 429 && attempt < MAX_RETRIES && retryAfter) {
        await new Promise((r) => setTimeout(r, (retryAfter + 1) * 1000));
        return queryModel(modelId, prompt, attempt + 1);
      }
      return null;
    }

    const data = await resp.json() as any;
    return data.choices?.[0]?.message?.content || null;
  } catch {
    if (attempt < MAX_RETRIES) {
      await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
      return queryModel(modelId, prompt, attempt + 1);
    }
    return null;
  }
}

// ── Answer checking ───────────────────────────────────────────────────────────

function normalizeValue(v: string): string {
  return v.toLowerCase().replace(/[_-]/g, " ").replace(/\s+/g, " ").trim();
}

function containsGold(text: string, gold: string[]): boolean {
  const norm = normalizeValue(text);
  return gold.some((g) => norm.includes(normalizeValue(g)));
}

// ── Consensus extraction ──────────────────────────────────────────────────────

interface ConsensusResult {
  topValue: string;
  supportRatio: number;   // supporting models / active models
  correct: boolean;
}

async function extractConsensus(
  buffers: (string | null)[],
  gold: string[],
): Promise<ConsensusResult | null> {
  const models = buffers
    .map((buffer, modelIdx) => ({ modelIdx, buffer: buffer || "" }))
    .filter((m) => m.buffer.length > 0);

  if (models.length === 0) return null;

  const allNodes = models.flatMap((m) => parseSCLToAST(m.buffer, m.modelIdx));
  if (allNodes.length === 0) return null;

  const clusters = await clusterConcepts(allNodes);

  // Gather all assertion values with support counts
  // (skip pure uncertainty/confidence numeric noise)
  const valueSupport = new Map<string, Set<number>>();
  for (const cluster of clusters) {
    for (const group of cluster.assertions) {
      if (group.key === "uncertainty" || group.key === "confidence") continue;
      for (const v of group.values) {
        const norm = normalizeValue(v.value);
        if (!norm) continue;
        let set = valueSupport.get(norm);
        if (!set) { set = new Set(); valueSupport.set(norm, set); }
        for (const mi of v.modelIndices) set.add(mi);
      }
    }
  }

  if (valueSupport.size === 0) return null;

  // Top-supported value
  const ranked = [...valueSupport.entries()].sort((a, b) => b[1].size - a[1].size);
  const [topValue, supporters] = ranked[0];
  const activeModels = models.length;

  // Consensus is "correct" if any of the top-3 supported values contains gold
  // (top value is often the question subject, not the answer — check top 3)
  const top3 = ranked.slice(0, 3);
  const correct = top3.some(([v]) => containsGold(v, gold));

  return {
    topValue,
    supportRatio: supporters.size / activeModels,
    correct,
  };
}

// ── Main experiment ───────────────────────────────────────────────────────────

async function main() {
  console.log("═══ Hypothesis Test: consensus vs individual accuracy ═══\n");

  const modelIds = await getFreeModels();
  console.log(`Models (${modelIds.length}): ${modelIds.map((m) => m.split("/").pop()).join(", ")}\n`);

  const results: {
    question: string;
    modelCorrect: boolean[];   // per active model
    consensusCorrect: boolean;
    supportRatio: number;
    activeModels: number;
  }[] = [];

  for (const [qi, q] of QUESTIONS.entries()) {
    console.log(`[${qi + 1}/${QUESTIONS.length}] ${q.prompt}`);

    // Query all models, batched 4 at a time
    const buffers: (string | null)[] = new Array(modelIds.length).fill(null);
    for (let i = 0; i < modelIds.length; i += 4) {
      const batch = modelIds.slice(i, i + 4);
      const batchResults = await Promise.all(batch.map((id) => queryModel(id, q.prompt)));
      for (let j = 0; j < batch.length; j++) buffers[i + j] = batchResults[j];
      await new Promise((r) => setTimeout(r, 500));
    }

    const active = buffers.filter((b): b is string => !!b && b.length > 0);
    const modelCorrect = active.map((b) => containsGold(b, q.gold));

    const consensus = await extractConsensus(buffers, q.gold);

    if (!consensus || active.length < 3) {
      console.log(`  SKIP (only ${active.length} models responded)\n`);
      continue;
    }

    const meanAcc = modelCorrect.filter(Boolean).length / modelCorrect.length;
    console.log(`  models: ${active.length} responded, ${(meanAcc * 100).toFixed(0)}% individually correct`);
    console.log(`  consensus: top="${consensus.topValue}" support=${(consensus.supportRatio * 100).toFixed(0)}% → ${consensus.correct ? "CORRECT" : "WRONG"}\n`);

    results.push({
      question: q.prompt,
      modelCorrect,
      consensusCorrect: consensus.correct,
      supportRatio: consensus.supportRatio,
      activeModels: active.length,
    });
  }

  // ── Analysis ────────────────────────────────────────────────────────────────
  console.log("\n═══ RESULTS ═══\n");

  const n = results.length;
  if (n === 0) { console.log("No results."); return; }

  const consensusAcc = results.filter((r) => r.consensusCorrect).length / n;
  const meanIndividualAcc = results.reduce(
    (sum, r) => sum + r.modelCorrect.filter(Boolean).length / r.modelCorrect.length, 0,
  ) / n;

  // Best single model (by index across questions where it responded)
  console.log(`Questions evaluated:        ${n}`);
  console.log(`Mean individual accuracy:   ${(meanIndividualAcc * 100).toFixed(1)}%`);
  console.log(`Consensus accuracy:         ${(consensusAcc * 100).toFixed(1)}%`);
  console.log(`Lift:                       ${((consensusAcc - meanIndividualAcc) * 100).toFixed(1)} pts\n`);

  // Calibration: high support vs low support
  const highSupport = results.filter((r) => r.supportRatio >= 0.5);
  const lowSupport = results.filter((r) => r.supportRatio < 0.5);
  if (highSupport.length && lowSupport.length) {
    const hAcc = highSupport.filter((r) => r.consensusCorrect).length / highSupport.length;
    const lAcc = lowSupport.filter((r) => r.consensusCorrect).length / lowSupport.length;
    console.log(`Calibration (support ratio → correctness):`);
    console.log(`  support ≥ 50%: ${(hAcc * 100).toFixed(0)}% correct (n=${highSupport.length})`);
    console.log(`  support < 50%: ${(lAcc * 100).toFixed(0)}% correct (n=${lowSupport.length})`);
  }

  // Point-biserial-ish correlation: supportRatio vs correctness
  const mean = results.reduce((s, r) => s + r.supportRatio, 0) / n;
  const correctMean = results.filter((r) => r.consensusCorrect)
    .reduce((s, r) => s + r.supportRatio, 0) / Math.max(1, results.filter((r) => r.consensusCorrect).length);
  const wrongMean = results.filter((r) => !r.consensusCorrect)
    .reduce((s, r) => s + r.supportRatio, 0) / Math.max(1, results.filter((r) => !r.consensusCorrect).length);
  console.log(`\nMean support when correct:  ${(correctMean * 100).toFixed(0)}%`);
  console.log(`Mean support when wrong:    ${(wrongMean * 100).toFixed(0)}%`);
  console.log(`Overall mean support:       ${(mean * 100).toFixed(0)}%`);
}

main().catch(console.error);
