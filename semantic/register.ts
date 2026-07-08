// ═══════════════════════════════════════════════════════════════════════════════
// Spatial Registration — Concept Clusters → Pixel Grid
//
// Maps semantic clusters to deterministic grid regions:
//
//   same concept + same value → same pixels → glow by support count
//   same concept + different value → same region, different colored branch
//   different concept → different region
//
// The coordinate system is MEANING SPACE, not text space.
// ═══════════════════════════════════════════════════════════════════════════════

import type { ConceptCluster, AssertionGroup } from "./cluster.ts";

export interface RGB { r: number; g: number; b: number }

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SemanticPixel {
  row: number;
  col: number;
  char: string;
  rgb: RGB;
  modelIndices: number[];
  agreementRatio: number;  // how many models placed this vs total
  clusterId: string;
  assertionKey?: string;
  assertionValue?: string;
  isBranch: boolean;       // true if this is a divergent value branch
}

export interface SemanticGrid {
  rows: number;
  cols: number;
  pixels: SemanticPixel[];
  clusters: ConceptCluster[];
}

// ── Registration ──────────────────────────────────────────────────────────────

const REGION_WIDTH = 40;  // chars wide per concept region
const REGION_GAP = 2;     // blank rows between regions
const INDENT = 2;

export function registerToGrid(
  clusters: ConceptCluster[],
  totalModels: number,
): SemanticGrid {
  const pixels: SemanticPixel[] = [];
  let currentRow = 0;

  // Sort clusters: most models first (strongest consensus at top)
  const sorted = [...clusters].sort((a, b) => b.modelIndices.length - a.modelIndices.length);

  for (const cluster of sorted) {
    const regionPixels = renderClusterRegion(cluster, currentRow, totalModels);
    pixels.push(...regionPixels);

    // Calculate how many rows this region used
    const maxRow = regionPixels.reduce((max, p) => Math.max(max, p.row), currentRow);
    currentRow = maxRow + REGION_GAP + 1;
  }

  const maxRow = pixels.reduce((max, p) => Math.max(max, p.row), 0);
  const maxCol = pixels.reduce((max, p) => Math.max(max, p.col), 0);

  return {
    rows: maxRow + 1,
    cols: maxCol + 1,
    pixels,
    clusters: sorted,
  };
}

// ── Render a single cluster region ────────────────────────────────────────────

function renderClusterRegion(
  cluster: ConceptCluster,
  startRow: number,
  totalModels: number,
): SemanticPixel[] {
  const pixels: SemanticPixel[] = [];
  let row = startRow;

  // Header: @concept_name [models: N]
  const header = `@${cluster.normalizedName}`;
  const support = cluster.modelIndices.length;
  const headerRatio = support / totalModels;

  for (let i = 0; i < header.length; i++) {
    pixels.push({
      row, col: i, char: header[i],
      rgb: consensusColor(headerRatio, cluster.modelIndices, totalModels),
      modelIndices: cluster.modelIndices,
      agreementRatio: headerRatio,
      clusterId: cluster.id,
      isBranch: false,
    });
  }

  // Match type indicator
  const indicator = cluster.matchType === "exact" ? " ≡" :
                    cluster.matchType === "alias" ? " ≈" : " ~";
  const supportStr = `${indicator} [${support}/${totalModels}]`;
  for (let i = 0; i < supportStr.length; i++) {
    pixels.push({
      row, col: header.length + i, char: supportStr[i],
      rgb: { r: 100, g: 100, b: 100 },
      modelIndices: cluster.modelIndices,
      agreementRatio: headerRatio,
      clusterId: cluster.id,
      isBranch: false,
    });
  }
  row++;

  // Assertions: grouped by key, values as branches
  for (const group of cluster.assertions) {
    row = renderAssertionGroup(group, cluster, row, totalModels, pixels);
  }

  return pixels;
}

function renderAssertionGroup(
  group: AssertionGroup,
  cluster: ConceptCluster,
  startRow: number,
  totalModels: number,
  pixels: SemanticPixel[],
): number {
  let row = startRow;

  if (group.values.length === 1) {
    // Single value — all models agree (or only one model asserted)
    const { value, modelIndices } = group.values[0];
    const line = `  ${group.key} → ${value}`;
    const ratio = modelIndices.length / totalModels;

    for (let col = 0; col < line.length && col < REGION_WIDTH; col++) {
      if (line[col] === " ") continue; // skip spaces
      pixels.push({
        row, col, char: line[col],
        rgb: consensusColor(ratio, modelIndices, totalModels),
        modelIndices,
        agreementRatio: ratio,
        clusterId: cluster.id,
        assertionKey: group.key,
        assertionValue: value,
        isBranch: false,
      });
    }
    row++;
  } else {
    // Multiple values — BRANCH: same key, divergent values
    // Show the key, then branch for each value
    const keyLine = `  ${group.key} →`;
    const allModels = group.values.flatMap((v) => v.modelIndices);
    const keyRatio = allModels.length / totalModels;

    for (let col = 0; col < keyLine.length; col++) {
      if (keyLine[col] === " ") continue;
      pixels.push({
        row, col, char: keyLine[col],
        rgb: consensusColor(keyRatio, allModels, totalModels),
        modelIndices: allModels,
        agreementRatio: keyRatio,
        clusterId: cluster.id,
        assertionKey: group.key,
        isBranch: false,
      });
    }
    row++;

    // Each value gets its own branch line
    const sortedValues = [...group.values].sort((a, b) => b.modelIndices.length - a.modelIndices.length);
    for (const { value, modelIndices } of sortedValues) {
      const branchLine = `    ├ ${value}`;
      const ratio = modelIndices.length / totalModels;

      for (let col = 0; col < branchLine.length && col < REGION_WIDTH; col++) {
        if (branchLine[col] === " ") continue;
        pixels.push({
          row, col, char: branchLine[col],
          rgb: consensusColor(ratio, modelIndices, totalModels),
          modelIndices,
          agreementRatio: ratio,
          clusterId: cluster.id,
          assertionKey: group.key,
          assertionValue: value,
          isBranch: true,
        });
      }
      row++;
    }
  }

  return row;
}

// ── Color Logic ───────────────────────────────────────────────────────────────

const MODEL_COLORS: RGB[] = [
  { r: 231, g: 76, b: 60 }, { r: 52, g: 152, b: 219 }, { r: 46, g: 204, b: 113 },
  { r: 241, g: 196, b: 15 }, { r: 155, g: 89, b: 182 }, { r: 230, g: 126, b: 34 },
  { r: 26, g: 188, b: 156 }, { r: 232, g: 67, b: 147 }, { r: 86, g: 204, b: 242 },
  { r: 253, g: 203, b: 110 }, { r: 108, g: 239, b: 177 }, { r: 223, g: 130, b: 201 },
  { r: 99, g: 205, b: 218 }, { r: 255, g: 168, b: 120 }, { r: 162, g: 155, b: 254 },
  { r: 129, g: 236, b: 104 }, { r: 255, g: 107, b: 107 }, { r: 100, g: 181, b: 246 },
  { r: 178, g: 235, b: 109 }, { r: 239, g: 131, b: 190 }, { r: 72, g: 219, b: 200 },
  { r: 255, g: 183, b: 77 }, { r: 149, g: 175, b: 242 }, { r: 216, g: 216, b: 78 },
];

function modelColor(idx: number): RGB {
  return MODEL_COLORS[idx % MODEL_COLORS.length];
}

function consensusColor(_ratio: number, modelIndices: number[], totalModels: number): RGB {
  if (modelIndices.length === 0) return { r: 20, g: 20, b: 25 };
  if (modelIndices.length === totalModels && totalModels > 1) return { r: 255, g: 255, b: 255 };
  if (modelIndices.length === 1) return modelColor(modelIndices[0]);

  const ratio = modelIndices.length / totalModels;
  let mr = 1, mg = 1, mb = 1;
  for (const idx of modelIndices) {
    const c = modelColor(idx);
    mr *= c.r / 255;
    mg *= c.g / 255;
    mb *= c.b / 255;
  }
  const lift = ratio * 0.7;
  return {
    r: Math.round(Math.min(255, (mr + lift) * 255)),
    g: Math.round(Math.min(255, (mg + lift) * 255)),
    b: Math.round(Math.min(255, (mb + lift) * 255)),
  };
}
