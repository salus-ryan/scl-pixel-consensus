// ═══════════════════════════════════════════════════════════════════════════════
// Semantic Concept Clustering
//
// Pipeline:
//   1. Exact match on normalized concept name
//   2. Alias/normalization match (cheap, deterministic)
//   3. Local embedding similarity (for unresolved pairs)
//   4. Remaining unresolved → separate clusters
//
// Output: ConceptCluster[] — each cluster is a group of semantically
// equivalent concepts from different models.
//
// CRITICAL: Values are NOT merged by embedding. "Trump" and "Biden" may be
// close in vector space but are contradictory assertions. Values use exact
// identity comparison only.
// ═══════════════════════════════════════════════════════════════════════════════

import { type SCLConceptNode, type SCLAssertion, cheapMatch } from "./normalize.ts";
import { embedBatch, cosineSim } from "./embeddings.ts";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface AssertionGroup {
  key: string;              // normalized key
  values: {
    value: string;          // exact value text
    modelIndices: number[]; // which models assert this value
  }[];
}

export interface ConceptCluster {
  id: string;                   // cluster identifier (canonical name)
  conceptNames: string[];       // all raw names that mapped here
  normalizedName: string;       // the canonical normalized name
  matchType: "exact" | "alias" | "embedding";
  similarity: number;           // min similarity within cluster
  modelIndices: number[];       // all contributing models
  assertions: AssertionGroup[]; // grouped by key, split by value
  confidence: number;           // average confidence across models
}

// ── Clustering Pipeline ───────────────────────────────────────────────────────

const EMBEDDING_THRESHOLD = 0.65; // cosine sim threshold for concept clustering

export async function clusterConcepts(
  allNodes: SCLConceptNode[],
): Promise<ConceptCluster[]> {
  if (allNodes.length === 0) return [];

  // Step 1+2: Group by exact/alias match
  const clusters: {
    normalizedName: string;
    nodes: SCLConceptNode[];
    matchType: "exact" | "alias" | "embedding";
    similarity: number;
  }[] = [];

  const assigned = new Set<number>(); // indices into allNodes

  for (let i = 0; i < allNodes.length; i++) {
    if (assigned.has(i)) continue;

    const cluster: {
      normalizedName: string;
      nodes: SCLConceptNode[];
      matchType: "exact" | "alias" | "embedding";
      similarity: number;
    } = {
      normalizedName: allNodes[i].name,
      nodes: [allNodes[i]],
      matchType: "exact",
      similarity: 1.0,
    };
    assigned.add(i);

    for (let j = i + 1; j < allNodes.length; j++) {
      if (assigned.has(j)) continue;

      const match = cheapMatch(allNodes[i].name, allNodes[j].name);
      if (match) {
        cluster.nodes.push(allNodes[j]);
        cluster.matchType = match.type === "exact" ? cluster.matchType : "alias";
        cluster.similarity = Math.min(cluster.similarity, match.similarity);
        assigned.add(j);
      }
    }

    clusters.push(cluster);
  }

  // Step 3: Embedding pass for unresolved pairs (single-node clusters)
  const singleClusters = clusters.filter((c) => c.nodes.length === 1);
  const multiClusters = clusters.filter((c) => c.nodes.length > 1);

  if (singleClusters.length >= 2) {
    // Embed all single-cluster concept names
    const textsToEmbed = singleClusters.map((c) => {
      // Embed a descriptive phrase, not just the snake_case name
      const name = c.normalizedName.replace(/_/g, " ");
      // Include first assertion key-values for context
      const context = c.nodes[0].assertions
        .slice(0, 3)
        .map((a) => `${a.key}: ${a.value}`)
        .join(", ");
      return context ? `${name} (${context})` : name;
    });

    const embeddings = await embedBatch(textsToEmbed);

    // Greedy clustering by cosine similarity
    const embeddingAssigned = new Set<number>();

    for (let i = 0; i < singleClusters.length; i++) {
      if (embeddingAssigned.has(i)) continue;
      embeddingAssigned.add(i);

      const merged: number[] = [i];

      for (let j = i + 1; j < singleClusters.length; j++) {
        if (embeddingAssigned.has(j)) continue;

        const sim = cosineSim(embeddings[i], embeddings[j]);
        if (sim >= EMBEDDING_THRESHOLD) {
          merged.push(j);
          embeddingAssigned.add(j);
        }
      }

      if (merged.length > 1) {
        // Merge into one cluster
        const mergedCluster = {
          normalizedName: singleClusters[merged[0]].normalizedName,
          nodes: merged.flatMap((idx) => singleClusters[idx].nodes),
          matchType: "embedding" as const,
          similarity: Math.min(
            ...merged.slice(1).map((idx) => cosineSim(embeddings[merged[0]], embeddings[idx])),
          ),
        };
        multiClusters.push(mergedCluster);
      } else {
        // Remains single
        multiClusters.push(singleClusters[i]);
      }
    }
  } else {
    multiClusters.push(...singleClusters);
  }

  // Step 4: Build final ConceptCluster output with assertion grouping
  return multiClusters.map((c) => buildCluster(c));
}

// ── Build final cluster with assertion groups ─────────────────────────────────

function buildCluster(raw: {
  normalizedName: string;
  nodes: SCLConceptNode[];
  matchType: "exact" | "alias" | "embedding";
  similarity: number;
}): ConceptCluster {
  // Group assertions by normalized key
  const keyGroups = new Map<string, Map<string, number[]>>(); // key → value → modelIndices

  for (const node of raw.nodes) {
    for (const assertion of node.assertions) {
      let valueMap = keyGroups.get(assertion.key);
      if (!valueMap) {
        valueMap = new Map();
        keyGroups.set(assertion.key, valueMap);
      }

      const models = valueMap.get(assertion.value) || [];
      models.push(node.modelIdx);
      valueMap.set(assertion.value, models);
    }
  }

  const assertions: AssertionGroup[] = [];
  for (const [key, valueMap] of keyGroups) {
    const values = [...valueMap.entries()].map(([value, modelIndices]) => ({
      value,
      modelIndices,
    }));
    assertions.push({ key, values });
  }

  const modelIndices = [...new Set(raw.nodes.map((n) => n.modelIdx))];
  const confidence = raw.nodes.reduce((sum, n) => sum + n.confidence, 0) / raw.nodes.length;

  return {
    id: raw.normalizedName,
    conceptNames: [...new Set(raw.nodes.map((n) => n.rawName))],
    normalizedName: raw.normalizedName,
    matchType: raw.matchType,
    similarity: raw.similarity,
    modelIndices,
    assertions,
    confidence,
  };
}
