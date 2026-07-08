// SCL parser + pixel logic (shared with backend extension)

export interface SCLConcept {
  name: string;
  props: Map<string, string>;
  confidence: number;
}

export function parseSCL(text: string): SCLConcept[] {
  const concepts: SCLConcept[] = [];
  const blockRe = /@(\w+)\s*\[([\s\S]*?)\]/g;
  let match: RegExpExecArray | null;

  while ((match = blockRe.exec(text)) !== null) {
    const name = match[1]!;
    const body = match[2]!;
    const props = new Map<string, string>();
    let confidence = 0.5;

    for (const line of body.split("\n")) {
      const clean = line.trim();
      if (!clean || !clean.includes("→")) continue;
      const [k, ...rest] = clean.split("→");
      const key = k!.trim();
      const val = rest.join("→").trim();
      props.set(key, val);
      if (key === "uncertainty") {
        const n = parseFloat(val);
        if (!Number.isNaN(n)) confidence = Math.min(1, Math.max(0, n));
      }
    }

    concepts.push({ name, props, confidence });
  }

  return concepts;
}

export function canonicalLayout(concepts: SCLConcept[]): string[] {
  if (concepts.length === 0) return [];
  const sorted = [...concepts].sort((a, b) => a.name.localeCompare(b.name));
  const lines: string[] = [];

  for (const c of sorted) {
    lines.push(`@${c.name} [`);
    const entries = [...c.props.entries()];
    const regular = entries.filter(([k]) => k !== "uncertainty").sort((a, b) => a[0].localeCompare(b[0]));
    const uncertainty = entries.find(([k]) => k === "uncertainty");

    for (const [k, v] of regular) {
      lines.push(`  ${k} → ${v}`);
    }
    if (uncertainty) {
      lines.push(`  uncertainty → ${uncertainty[1]}`);
    }
    lines.push(`]`);
    lines.push(``);
  }

  return lines;
}

export type PixelLayer = Map<number, Map<number, string>>;

export function textToPixelLayer(lines: string[]): PixelLayer {
  const layer: PixelLayer = new Map();
  for (let row = 0; row < lines.length; row++) {
    const line = lines[row];
    if (line.length === 0) continue;
    const rowMap = new Map<number, string>();
    for (let col = 0; col < line.length; col++) {
      if (line[col] !== " ") {
        rowMap.set(col, line[col]);
      }
    }
    if (rowMap.size > 0) layer.set(row, rowMap);
  }
  return layer;
}

export interface RGB { r: number; g: number; b: number }

export const MODEL_COLORS: RGB[] = [
  { r: 231, g: 76, b: 60 },
  { r: 52, g: 152, b: 219 },
  { r: 46, g: 204, b: 113 },
  { r: 241, g: 196, b: 15 },
  { r: 155, g: 89, b: 182 },
  { r: 230, g: 126, b: 34 },
  { r: 26, g: 188, b: 156 },
  { r: 232, g: 67, b: 147 },
  { r: 86, g: 204, b: 242 },
  { r: 253, g: 203, b: 110 },
  { r: 108, g: 239, b: 177 },
  { r: 223, g: 130, b: 201 },
  { r: 99, g: 205, b: 218 },
  { r: 255, g: 168, b: 120 },
  { r: 162, g: 155, b: 254 },
  { r: 129, g: 236, b: 104 },
  { r: 255, g: 107, b: 107 },
  { r: 100, g: 181, b: 246 },
  { r: 178, g: 235, b: 109 },
  { r: 239, g: 131, b: 190 },
  { r: 72, g: 219, b: 200 },
  { r: 255, g: 183, b: 77 },
  { r: 149, g: 175, b: 242 },
  { r: 216, g: 216, b: 78 },
];

export function modelColor(idx: number): RGB {
  return MODEL_COLORS[idx % MODEL_COLORS.length];
}

export function compositeColors(modelIndices: number[], totalModels: number): RGB {
  if (modelIndices.length === 0) return { r: 20, g: 20, b: 25 };

  if (modelIndices.length === totalModels && totalModels > 1) {
    return { r: 255, g: 255, b: 255 };
  }

  if (modelIndices.length === 1) {
    return modelColor(modelIndices[0]);
  }

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

export interface CompositePixel {
  char: string;
  rgb: RGB;
  agreementRatio: number;
  modelIndices: number[];
}

export function compositeGrid(
  layers: { layer: PixelLayer; colorIdx: number }[],
  totalModels: number,
): { rows: number; cols: number; pixels: Map<string, CompositePixel> } {
  let maxRow = 0, maxCol = 0;
  for (const { layer } of layers) {
    for (const [row, cols] of layer) {
      if (row > maxRow) maxRow = row;
      for (const col of cols.keys()) {
        if (col > maxCol) maxCol = col;
      }
    }
  }

  const pixels = new Map<string, CompositePixel>();

  for (let row = 0; row <= maxRow; row++) {
    for (let col = 0; col <= maxCol; col++) {
      const contributions: { char: string; modelIdx: number }[] = [];

      for (const { layer, colorIdx } of layers) {
        const rowMap = layer.get(row);
        if (!rowMap) continue;
        const ch = rowMap.get(col);
        if (ch !== undefined) {
          contributions.push({ char: ch, modelIdx: colorIdx });
        }
      }

      if (contributions.length === 0) continue;

      const charGroups = new Map<string, number[]>();
      for (const { char, modelIdx } of contributions) {
        const arr = charGroups.get(char) || [];
        arr.push(modelIdx);
        charGroups.set(char, arr);
      }

      if (charGroups.size === 1) {
        const [char, indices] = [...charGroups.entries()][0];
        const rgb = compositeColors(indices, totalModels);
        pixels.set(`${row},${col}`, { char, rgb, agreementRatio: indices.length / totalModels, modelIndices: indices });
      } else {
        let bestChar = "";
        let bestCount = 0;
        let bestIndices: number[] = [];
        for (const [char, indices] of charGroups) {
          if (indices.length > bestCount || (indices.length === bestCount && char < bestChar)) {
            bestChar = char;
            bestCount = indices.length;
            bestIndices = indices;
          }
        }
        const allIndices = contributions.map((c) => c.modelIdx);
        const rgb = compositeColors(bestIndices, totalModels);
        const dimFactor = bestCount / contributions.length;
        pixels.set(`${row},${col}`, {
          char: bestChar,
          rgb: { r: Math.round(rgb.r * dimFactor), g: Math.round(rgb.g * dimFactor), b: Math.round(rgb.b * dimFactor) },
          agreementRatio: bestCount / totalModels,
          modelIndices: allIndices,
        });
      }
    }
  }

  return { rows: maxRow + 1, cols: maxCol + 1, pixels };
}
