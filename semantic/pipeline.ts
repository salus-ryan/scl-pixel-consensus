// ═══════════════════════════════════════════════════════════════════════════════
// Semantic Pipeline — full path from raw SCL buffers to semantic pixel grid
//
// SCL → parse AST → normalize → cluster → register → pixels
// ═══════════════════════════════════════════════════════════════════════════════

import { parseSCLToAST, type SCLConceptNode } from "./normalize.ts";
import { clusterConcepts, type ConceptCluster } from "./cluster.ts";
import { registerToGrid, type SemanticGrid, type SemanticPixel } from "./register.ts";

export type { SemanticGrid, SemanticPixel, ConceptCluster };

export interface ModelBuffer {
  modelIdx: number;
  buffer: string;
  error?: string;
}

/**
 * Run the full semantic pipeline on current model buffers.
 * Returns a SemanticGrid ready for rendering.
 */
export async function runSemanticPipeline(
  models: ModelBuffer[],
): Promise<SemanticGrid> {
  // Step 1: Parse all buffers to AST
  const allNodes: SCLConceptNode[] = [];
  for (const m of models) {
    if (m.error || !m.buffer) continue;
    const nodes = parseSCLToAST(m.buffer, m.modelIdx);
    allNodes.push(...nodes);
  }

  if (allNodes.length === 0) {
    return { rows: 0, cols: 0, pixels: [], clusters: [] };
  }

  // Step 2: Cluster concepts (exact → alias → embedding)
  const clusters = await clusterConcepts(allNodes);

  // Step 3: Register clusters to spatial grid
  const totalModels = models.filter((m) => !m.error).length;
  const grid = registerToGrid(clusters, totalModels);

  return grid;
}

/**
 * Serialize SemanticGrid for JSON transport to frontend.
 * (SemanticPixel[] is already plain objects, no Maps needed)
 */
export function serializeGrid(grid: SemanticGrid): any {
  return {
    rows: grid.rows,
    cols: grid.cols,
    pixels: grid.pixels,
    clusters: grid.clusters.map((c) => ({
      id: c.id,
      conceptNames: c.conceptNames,
      normalizedName: c.normalizedName,
      matchType: c.matchType,
      similarity: c.similarity,
      modelIndices: c.modelIndices,
      assertions: c.assertions,
      confidence: c.confidence,
    })),
  };
}
