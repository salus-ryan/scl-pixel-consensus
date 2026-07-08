// ═══════════════════════════════════════════════════════════════════════════════
// SCL AST + Deterministic Normalization
//
// Parse raw SCL text → typed AST, then normalize:
//   - lowercase all concept names
//   - collapse underscores/hyphens/camelCase to snake_case
//   - strip articles, filler
//   - known alias table for common synonyms
// ═══════════════════════════════════════════════════════════════════════════════

export interface SCLAssertion {
  key: string;         // normalized property name
  rawKey: string;      // original key text
  value: string;       // raw value (NOT normalized — values are identity-compared)
  relation?: string;   // if triple: subject → relation → object
}

export interface SCLConceptNode {
  name: string;        // normalized concept name
  rawName: string;     // original concept name
  assertions: SCLAssertion[];
  confidence: number;  // from uncertainty field
  modelIdx: number;    // which model produced this
}

// ── Parse ─────────────────────────────────────────────────────────────────────

export function parseSCLToAST(text: string, modelIdx: number): SCLConceptNode[] {
  const nodes: SCLConceptNode[] = [];
  const blockRe = /@(\w+)\s*\[([\s\S]*?)\]/g;
  let match: RegExpExecArray | null;

  while ((match = blockRe.exec(text)) !== null) {
    const rawName = match[1]!;
    const body = match[2]!;
    const assertions: SCLAssertion[] = [];
    let confidence = 0.5;

    for (const line of body.split("\n")) {
      const clean = line.trim();
      if (!clean || !clean.includes("→")) continue;

      const parts = clean.split("→").map((s) => s.trim());

      if (parts.length === 3) {
        // Triple: subject → relation → object
        assertions.push({
          key: normalizeKey(parts[0]),
          rawKey: parts[0],
          value: parts[2],
          relation: parts[1],
        });
      } else if (parts.length >= 2) {
        const key = parts[0];
        const val = parts.slice(1).join("→").trim();
        if (normalizeKey(key) === "uncertainty") {
          const n = parseFloat(val);
          if (!Number.isNaN(n)) confidence = Math.min(1, Math.max(0, n));
        } else {
          assertions.push({
            key: normalizeKey(key),
            rawKey: key,
            value: val,
          });
        }
      }
    }

    nodes.push({
      name: normalizeConcept(rawName),
      rawName,
      assertions,
      confidence,
      modelIdx,
    });
  }

  return nodes;
}

// ── Normalization ─────────────────────────────────────────────────────────────

const FILLER_WORDS = new Set([
  "the", "a", "an", "of", "in", "on", "at", "to", "for", "is", "are", "was",
  "were", "be", "been", "being", "has", "have", "had", "do", "does", "did",
  "this", "that", "these", "those", "it", "its",
]);

/** Normalize a concept name to canonical form */
export function normalizeConcept(raw: string): string {
  // camelCase → snake_case
  let s = raw.replace(/([a-z])([A-Z])/g, "$1_$2");
  // hyphens → underscores
  s = s.replace(/[-\s]+/g, "_");
  // lowercase
  s = s.toLowerCase();
  // strip filler words
  const parts = s.split("_").filter((p) => p.length > 0 && !FILLER_WORDS.has(p));
  s = parts.join("_");
  // Apply known aliases
  s = CONCEPT_ALIASES.get(s) || s;
  return s;
}

/** Normalize a property key */
export function normalizeKey(raw: string): string {
  let s = raw.replace(/([a-z])([A-Z])/g, "$1_$2");
  s = s.replace(/[-\s]+/g, "_").toLowerCase().trim();
  const parts = s.split("_").filter((p) => p.length > 0 && !FILLER_WORDS.has(p));
  s = parts.join("_");
  return KEY_ALIASES.get(s) || s;
}

// ── Alias Tables ──────────────────────────────────────────────────────────────
// These are cheap deterministic merges that don't need embeddings.

const CONCEPT_ALIASES = new Map<string, string>([
  // presidency — all resolve to head_of_state
  ["president_united_states", "head_of_state"],
  ["us_president", "head_of_state"],
  ["president_us", "head_of_state"],
  ["potus", "head_of_state"],
  ["current_president", "head_of_state"],
  ["president", "head_of_state"],
  ["head_state", "head_of_state"],
  ["leader", "head_of_state"],
  ["us_leader", "head_of_state"],
  ["current_leader", "head_of_state"],
  // general
  ["info", "information"],
  ["knowledge", "information"],
  ["facts", "information"],
  ["fact", "information"],
  ["answer", "response"],
  ["reply", "response"],
  ["query", "question"],
  ["inquiry", "question"],
]);

const KEY_ALIASES = new Map<string, string>([
  ["person", "holder"],
  ["name", "holder"],
  ["individual", "holder"],
  ["who", "holder"],
  ["country", "nation"],
  ["state", "nation"],
  ["domain", "field"],
  ["area", "field"],
  ["topic", "field"],
  ["temp", "temperature"],
]);

// ── Exact + Alias matching (first pass before embeddings) ─────────────────────

export interface ConceptMatch {
  type: "exact" | "alias" | "embedding" | "unresolved";
  similarity: number; // 1.0 for exact, 0.9 for alias, cosine for embedding
}

/** Check if two normalized concept names match via exact or alias */
export function cheapMatch(a: string, b: string): ConceptMatch | null {
  if (a === b) return { type: "exact", similarity: 1.0 };

  // Check if they share a common alias target
  const aliasA = CONCEPT_ALIASES.get(a);
  const aliasB = CONCEPT_ALIASES.get(b);
  if (aliasA && aliasA === b) return { type: "alias", similarity: 0.95 };
  if (aliasB && aliasB === a) return { type: "alias", similarity: 0.95 };
  if (aliasA && aliasB && aliasA === aliasB) return { type: "alias", similarity: 0.9 };

  // Token overlap heuristic (quick pre-filter for embedding candidates)
  const tokensA = new Set(a.split("_"));
  const tokensB = new Set(b.split("_"));
  const intersection = [...tokensA].filter((t) => tokensB.has(t));
  const union = new Set([...tokensA, ...tokensB]);
  const jaccard = intersection.length / union.size;
  if (jaccard >= 0.5) return { type: "alias", similarity: 0.7 + jaccard * 0.2 };

  return null; // needs embedding
}
