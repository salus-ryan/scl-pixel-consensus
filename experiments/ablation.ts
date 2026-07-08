// ═══════════════════════════════════════════════════════════════════════════════
// Ablation: does SCL per-assertion consensus beat free-text voting?
//
// Hard question set: misconception-loaded facts where individual models
// are expected to split (40-70% individual accuracy).
//
// Three arms, same models, same questions:
//   Arm A: SCL format → semantic pipeline → per-assertion consensus
//   Arm B: free text → local embeddings → cluster answers → majority cluster
//   Arm C: free text → normalized string majority vote
//
// SCL earns its keep only if A > B and A > C.
//
// Run: npx tsx experiments/ablation.ts
// ═══════════════════════════════════════════════════════════════════════════════

import dotenv from "dotenv";
import { parseSCLToAST } from "../semantic/normalize.ts";
import { clusterConcepts } from "../semantic/cluster.ts";
import { embedBatch, cosineSim } from "../semantic/embeddings.ts";

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

const PLAIN_SYSTEM = `Answer the question with a short, direct answer. One phrase or sentence maximum. No explanations.`;

// ── Hard question set: misconception-loaded ───────────────────────────────────

interface Question {
  prompt: string;
  gold: string[];
  trap?: string; // the common wrong answer
}

const QUESTIONS: Question[] = [
  { prompt: "What is the capital of Myanmar?", gold: ["naypyidaw", "nay pyi taw", "naypyitaw"], trap: "yangon" },
  { prompt: "What is the capital city of Canada?", gold: ["ottawa"], trap: "toronto" },
  { prompt: "What is the capital of Turkey?", gold: ["ankara"], trap: "istanbul" },
  { prompt: "What is the capital of Switzerland?", gold: ["bern"], trap: "zurich" },
  { prompt: "What is the capital of Brazil?", gold: ["brasilia", "brasília"], trap: "rio" },
  { prompt: "What is the largest desert in the world?", gold: ["antarctic", "antarctica"], trap: "sahara" },
  { prompt: "What is the national animal of Scotland?", gold: ["unicorn"], trap: "lion" },
  { prompt: "Who was the first person to win two Nobel Prizes?", gold: ["curie"], trap: "einstein" },
  { prompt: "How many hearts does an octopus have?", gold: ["3", "three"], trap: "1" },
  { prompt: "What is the smallest country in the world by area?", gold: ["vatican"], trap: "monaco" },
  { prompt: "Who invented the World Wide Web?", gold: ["berners-lee", "berners lee", "berners_lee"], trap: "internet" },
  { prompt: "Which country has the longest coastline in the world?", gold: ["canada"], trap: "russia" },
  { prompt: "Which planet in our solar system has the most confirmed moons?", gold: ["saturn"], trap: "jupiter" },
  { prompt: "In what year did the Berlin Wall fall?", gold: ["1989"], trap: "1991" },
];

// ── Shared helpers ────────────────────────────────────────────────────────────

async function getFreeModels(): Promise<string[]> {
  const resp = await fetch("https://openrouter.ai/api/v1/models", {
    headers: { Authorization: `Bearer ${OPENROUTER_API_KEY}` },
  });
  const data = await resp.json() as any;
  return (data.data || [])
    .filter((m: any) => m.id?.includes(":free") || m.pricing?.prompt === "0")
    .map((m: any) => m.id)
    .filter((id: string) => !id.includes("lyria"))
    .slice(0, 10);
}

async function queryModel(modelId: string, system: string, prompt: string, attempt = 0): Promise<string | null> {
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
          { role: "system", content: system },
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
        return queryModel(modelId, system, prompt, attempt + 1);
      }
      return null;
    }

    const data = await resp.json() as any;
    return data.choices?.[0]?.message?.content || null;
  } catch {
    if (attempt < MAX_RETRIES) {
      await new Promise((r) => setTimeout(r, 2000 * (attempt + 1)));
      return queryModel(modelId, system, prompt, attempt + 1);
    }
    return null;
  }
}

async function queryAllBatched(modelIds: string[], system: string, prompt: string): Promise<(string | null)[]> {
  const buffers: (string | null)[] = new Array(modelIds.length).fill(null);
  for (let i = 0; i < modelIds.length; i += 4) {
    const batch = modelIds.slice(i, i + 4);
    const results = await Promise.all(batch.map((id) => queryModel(id, system, prompt)));
    for (let j = 0; j < batch.length; j++) buffers[i + j] = results[j];
    await new Promise((r) => setTimeout(r, 500));
  }
  return buffers;
}

function normalizeValue(v: string): string {
  return v.toLowerCase().replace(/[_-]/g, " ").replace(/[.,!?"']/g, "").replace(/\s+/g, " ").trim();
}

function containsGold(text: string, gold: string[]): boolean {
  const norm = normalizeValue(text);
  return gold.some((g) => norm.includes(normalizeValue(g)));
}

// ── Arm A: SCL per-assertion consensus ────────────────────────────────────────

async function armA(buffers: (string | null)[], gold: string[]): Promise<{ correct: boolean; support: number } | null> {
  const models = buffers
    .map((buffer, modelIdx) => ({ modelIdx, buffer: buffer || "" }))
    .filter((m) => m.buffer.length > 0);
  if (models.length === 0) return null;

  const allNodes = models.flatMap((m) => parseSCLToAST(m.buffer, m.modelIdx));
  if (allNodes.length === 0) return null;

  const clusters = await clusterConcepts(allNodes);

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

  const ranked = [...valueSupport.entries()].sort((a, b) => b[1].size - a[1].size);
  const top3 = ranked.slice(0, 3);
  const correct = top3.some(([v]) => containsGold(v, gold));
  return { correct, support: ranked[0][1].size / models.length };
}

// ── Arm B: free text → embedding cluster → majority cluster ──────────────────

async function armB(answers: (string | null)[], gold: string[]): Promise<{ correct: boolean; support: number } | null> {
  const valid = answers.filter((a): a is string => !!a && a.trim().length > 0);
  if (valid.length === 0) return null;

  const embeddings = await embedBatch(valid.map((a) => normalizeValue(a)));

  // Greedy clustering at cosine 0.75
  const THRESHOLD = 0.75;
  const clusters: number[][] = [];
  const assigned = new Set<number>();
  for (let i = 0; i < valid.length; i++) {
    if (assigned.has(i)) continue;
    const cluster = [i];
    assigned.add(i);
    for (let j = i + 1; j < valid.length; j++) {
      if (assigned.has(j)) continue;
      if (cosineSim(embeddings[i], embeddings[j]) >= THRESHOLD) {
        cluster.push(j);
        assigned.add(j);
      }
    }
    clusters.push(cluster);
  }

  // Majority cluster
  clusters.sort((a, b) => b.length - a.length);
  const majority = clusters[0];
  // Correct if majority cluster's members contain gold
  const correct = majority.some((idx) => containsGold(valid[idx], gold));
  return { correct, support: majority.length / valid.length };
}

// ── Arm C: free text → normalized string majority vote ───────────────────────

function armC(answers: (string | null)[], gold: string[]): { correct: boolean; support: number } | null {
  const valid = answers.filter((a): a is string => !!a && a.trim().length > 0);
  if (valid.length === 0) return null;

  const counts = new Map<string, number>();
  for (const a of valid) {
    const norm = normalizeValue(a);
    counts.set(norm, (counts.get(norm) || 0) + 1);
  }
  const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const [topAnswer, count] = ranked[0];
  const correct = containsGold(topAnswer, gold);
  return { correct, support: count / valid.length };
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log("═══ Ablation: SCL consensus vs free-text voting (hard set) ═══\n");

  const modelIds = await getFreeModels();
  console.log(`Models (${modelIds.length}): ${modelIds.map((m) => m.split("/").pop()).join(", ")}\n`);

  const results: {
    question: string;
    individualAcc: number;      // on plain-text arm
    a: { correct: boolean; support: number } | null;
    b: { correct: boolean; support: number } | null;
    c: { correct: boolean; support: number } | null;
  }[] = [];

  for (const [qi, q] of QUESTIONS.entries()) {
    console.log(`[${qi + 1}/${QUESTIONS.length}] ${q.prompt}`);

    // Query both arms
    const sclBuffers = await queryAllBatched(modelIds, SCL_SYSTEM, q.prompt);
    const plainAnswers = await queryAllBatched(modelIds, PLAIN_SYSTEM, q.prompt);

    const plainValid = plainAnswers.filter((a): a is string => !!a && a.trim().length > 0);
    const individualAcc = plainValid.length > 0
      ? plainValid.filter((a) => containsGold(a, q.gold)).length / plainValid.length
      : 0;

    const a = await armA(sclBuffers, q.gold);
    const b = await armB(plainAnswers, q.gold);
    const c = armC(plainAnswers, q.gold);

    console.log(`  individual (plain): ${(individualAcc * 100).toFixed(0)}% of ${plainValid.length} models`);
    console.log(`  A (SCL semantic):   ${a ? (a.correct ? "CORRECT" : "WRONG") + ` support=${(a.support * 100).toFixed(0)}%` : "N/A"}`);
    console.log(`  B (embed cluster):  ${b ? (b.correct ? "CORRECT" : "WRONG") + ` support=${(b.support * 100).toFixed(0)}%` : "N/A"}`);
    console.log(`  C (string vote):    ${c ? (c.correct ? "CORRECT" : "WRONG") + ` support=${(c.support * 100).toFixed(0)}%` : "N/A"}\n`);

    results.push({ question: q.prompt, individualAcc, a, b, c });
  }

  // ── Analysis ────────────────────────────────────────────────────────────────
  console.log("\n═══ RESULTS ═══\n");

  const n = results.length;
  const evaluated = results.filter((r) => r.a && r.b && r.c);
  const ne = evaluated.length;

  const meanIndividual = results.reduce((s, r) => s + r.individualAcc, 0) / n;
  const accA = evaluated.filter((r) => r.a!.correct).length / ne;
  const accB = evaluated.filter((r) => r.b!.correct).length / ne;
  const accC = evaluated.filter((r) => r.c!.correct).length / ne;

  console.log(`Questions evaluated:          ${ne}/${n}`);
  console.log(`Mean individual accuracy:     ${(meanIndividual * 100).toFixed(1)}%`);
  console.log(`Arm A (SCL semantic):         ${(accA * 100).toFixed(1)}%`);
  console.log(`Arm B (embedding cluster):    ${(accB * 100).toFixed(1)}%`);
  console.log(`Arm C (string majority):      ${(accC * 100).toFixed(1)}%\n`);

  // Per-question disagreements between arms
  console.log("Arm disagreements:");
  for (const r of evaluated) {
    const marks = `A=${r.a!.correct ? "✓" : "✗"} B=${r.b!.correct ? "✓" : "✗"} C=${r.c!.correct ? "✓" : "✗"}`;
    if (!(r.a!.correct && r.b!.correct && r.c!.correct)) {
      console.log(`  [${marks}] ${r.question}`);
    }
  }

  // Calibration on arm A
  const correctA = evaluated.filter((r) => r.a!.correct);
  const wrongA = evaluated.filter((r) => !r.a!.correct);
  if (correctA.length && wrongA.length) {
    const cs = correctA.reduce((s, r) => s + r.a!.support, 0) / correctA.length;
    const ws = wrongA.reduce((s, r) => s + r.a!.support, 0) / wrongA.length;
    console.log(`\nArm A calibration:`);
    console.log(`  mean support when correct: ${(cs * 100).toFixed(0)}%`);
    console.log(`  mean support when wrong:   ${(ws * 100).toFixed(0)}%`);
  }
}

main().catch(console.error);
