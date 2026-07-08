import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { Component, TUI } from "@earendil-works/pi-tui";

// ═══════════════════════════════════════════════════════════════════════════════
// SCL Pixel Consensus — visual ensemble via character-level registration
//
// Each model's SCL response is rendered into a canonical pixel grid
// (row, col → character). All grids are overlaid with subtractive compositing:
//
//   unanimous agreement → white-on-black (collapsed consensus)
//   strong agreement    → dark mixed color
//   minority claim      → bright source-model color
//   disagreement        → split/conflicting colors visible
//   empty               → no model asserted anything
//
// The crucial property: identical canonical SCL → identical pixels → overlap
// ═══════════════════════════════════════════════════════════════════════════════

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SCLConcept {
  name: string;
  props: Map<string, string>;
  confidenceSum: number;
  count: number;
}

export interface ModelSCL {
  modelId: string;
  concepts: SCLConcept[];
  raw: string;
  error?: string;
}

interface ModelStream {
  modelId: string;
  buffer: string;
  concepts: SCLConcept[];
  done: boolean;
  error?: string;
  tokenCount: number;
  colorIdx: number;
}

/** A single pixel in a model's SCL layer */
interface Pixel {
  char: string;           // the glyph at this position
  modelIndices: number[]; // which models placed this exact char here
}

/** The composited pixel grid — the merge IS the image */
type PixelGrid = Map<string, Pixel[]>; // key = "row,col" → overlapping pixels

// ── Model color palette (RGB for true-color compositing) ──────────────────────

interface RGB { r: number; g: number; b: number }

const MODEL_COLORS: RGB[] = [
  { r: 231, g:  76, b:  60 },  // red
  { r:  52, g: 152, b: 219 },  // blue
  { r:  46, g: 204, b: 113 },  // green
  { r: 241, g: 196, b:  15 },  // yellow
  { r: 155, g:  89, b: 182 },  // purple
  { r: 230, g: 126, b:  34 },  // orange
  { r:  26, g: 188, b: 156 },  // teal
  { r: 232, g:  67, b: 147 },  // pink
  { r:  86, g: 204, b: 242 },  // sky
  { r: 253, g: 203, b: 110 },  // peach
  { r: 108, g: 239, b: 177 },  // mint
  { r: 223, g: 130, b: 201 },  // orchid
  { r:  99, g: 205, b: 218 },  // cyan
  { r: 255, g: 168, b: 120 },  // salmon
  { r: 162, g: 155, b: 254 },  // lavender
  { r: 129, g: 236, b: 104 },  // lime
  { r: 255, g: 107, b: 107 },  // coral
  { r: 100, g: 181, b: 246 },  // periwinkle
  { r: 178, g: 235, b: 109 },  // chartreuse
  { r: 239, g: 131, b: 190 },  // rose
  { r:  72, g: 219, b: 200 },  // aqua
  { r: 255, g: 183, b:  77 },  // amber
  { r: 149, g: 175, b: 242 },  // steel blue
  { r: 216, g: 216, b:  78 },  // olive
];

function modelRGB(idx: number): RGB {
  return MODEL_COLORS[idx % MODEL_COLORS.length];
}

/** Composite N model colors via subtractive mixing (multiply blend) */
function compositeColors(modelIndices: number[], totalModels: number): RGB {
  if (modelIndices.length === 0) return { r: 40, g: 40, b: 40 };

  // Unanimous → white (all models agree, collapsed to consensus)
  if (modelIndices.length === totalModels) {
    return { r: 255, g: 255, b: 255 };
  }

  const ratio = modelIndices.length / totalModels;

  if (modelIndices.length === 1) {
    // Single model → its color at full brightness
    const c = modelRGB(modelIndices[0]);
    return c;
  }

  // Multiple but not all → multiply-blend, then brighten by agreement ratio
  // Multiply blend: each channel = product of (channel/255) across all sources
  let mr = 1, mg = 1, mb = 1;
  for (const idx of modelIndices) {
    const c = modelRGB(idx);
    mr *= c.r / 255;
    mg *= c.g / 255;
    mb *= c.b / 255;
  }

  // Brighten toward white as agreement grows
  const lift = ratio * 0.7;
  const r = Math.round(Math.min(255, (mr + lift) * 255));
  const g = Math.round(Math.min(255, (mg + lift) * 255));
  const b = Math.round(Math.min(255, (mb + lift) * 255));

  return { r, g, b };
}

function ansiTrueColor(rgb: RGB, text: string): string {
  return `\x1b[38;2;${rgb.r};${rgb.g};${rgb.b}m${text}\x1b[0m`;
}

function ansiBold(text: string): string {
  return `\x1b[1m${text}\x1b[0m`;
}

function ansiDim(text: string): string {
  return `\x1b[2m${text}\x1b[0m`;
}

function ansiModelColor(idx: number, text: string): string {
  const c = modelRGB(idx);
  return ansiTrueColor(c, text);
}

// ── Extension ─────────────────────────────────────────────────────────────────

export default async function (pi: ExtensionAPI) {
  pi.registerCommand("or-scl-merge", {
    description:
      "Stream free OpenRouter models and overlay SCL responses via pixel-level consensus compositing",
    handler: async (args, ctx) => {
      const prompt = args.trim();
      if (!prompt) {
        ctx.ui.notify(
          "Usage: /or-scl-merge <prompt> — pixel consensus across free models",
          "warning",
        );
        return;
      }

      const allModels = await ctx.modelRegistry.getAvailable();
      const freeModels = allModels.filter(
        (m) =>
          m.provider === "openrouter" &&
          (m.name?.includes("[FREE]") || m.id?.includes(":free")),
      );

      if (freeModels.length === 0) {
        ctx.ui.notify("No free OpenRouter models found.", "error");
        return;
      }

      const streams: ModelStream[] = freeModels.map((m, i) => ({
        modelId: m.id,
        buffer: "",
        concepts: [],
        done: false,
        tokenCount: 0,
        colorIdx: i,
      }));

      await ctx.ui.custom<void>(
        (tui, theme, _kb, done) => {
          const comp = new SCLPixelComponent(tui, theme, done, streams, prompt);
          comp.start();
          return comp;
        },
        {
          overlay: true,
          overlayOptions: { anchor: "center", width: 92, maxHeight: 50 },
        },
      );
    },
  });
}

// ── SCL system prompt ─────────────────────────────────────────────────────────

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

// ── SCL parser ────────────────────────────────────────────────────────────────

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

    concepts.push({ name, props, confidenceSum: confidence, count: 1 });
  }

  return concepts;
}

function parseSCLIncremental(text: string): {
  complete: SCLConcept[];
  partial: string | null;
} {
  const complete = parseSCL(text);
  const lastAt = text.lastIndexOf("@");
  if (lastAt >= 0) {
    const tail = text.slice(lastAt);
    const ob = tail.indexOf("[");
    const cb = tail.lastIndexOf("]");
    if (ob >= 0 && (cb < 0 || cb < ob)) {
      const m = tail.match(/^@(\w+)/);
      return { complete, partial: m ? `@${m[1]} [${tail.slice(ob + 1)}` : null };
    }
  }
  return { complete, partial: null };
}

// ═══════════════════════════════════════════════════════════════════════════════
// CANONICAL LAYOUT ENGINE
//
// Every SCL response is rendered to the SAME pixel grid layout so identical
// semantics produce identical coordinates. The layout is deterministic:
//
//   @concept_name [
//     key → value
//     key → value
//     uncertainty → 0.XX
//   ]
//
// Concepts are sorted by name. Props sorted alphabetically (uncertainty last).
// This guarantees: same meaning → same canonical SCL → same pixels.
// ═══════════════════════════════════════════════════════════════════════════════

/** Render concepts to canonical text lines */
export function canonicalLayout(concepts: SCLConcept[]): string[] {
  if (concepts.length === 0) return [];

  // Sort concepts by name
  const sorted = [...concepts].sort((a, b) => a.name.localeCompare(b.name));
  const lines: string[] = [];

  for (const c of sorted) {
    lines.push(`@${c.name} [`);

    // Sort props alphabetically, uncertainty last
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
    lines.push(``); // blank separator
  }

  return lines;
}

/** A model's pixel layer: row → col → char */
export type PixelLayer = Map<number, Map<number, string>>;

/** Render canonical text lines to a pixel layer */
export function textToPixelLayer(lines: string[]): PixelLayer {
  const layer: PixelLayer = new Map();
  for (let row = 0; row < lines.length; row++) {
    const line = lines[row];
    if (line.length === 0) continue;
    const rowMap = new Map<number, string>();
    for (let col = 0; col < line.length; col++) {
      if (line[col] !== " ") { // spaces are empty, not pixels
        rowMap.set(col, line[col]);
      }
    }
    if (rowMap.size > 0) layer.set(row, rowMap);
  }
  return layer;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PIXEL COMPOSITOR
//
// Overlay all model layers. At each (row, col):
//   - Collect which models placed which char
//   - If all agree on same char → unanimous → white
//   - If k/N agree → subtractive color blend, brighter with more agreement
//   - If only 1 model → that model's raw color
//   - Empty → dim background
// ═══════════════════════════════════════════════════════════════════════════════

interface CompositePixel {
  char: string;
  rgb: RGB;
  agreementRatio: number;  // 0..1
  modelIndices: number[];
}

interface CompositeGrid {
  rows: number;
  cols: number;
  pixels: Map<string, CompositePixel>; // "row,col" → pixel
}

export function compositeLayersToGrid(
  layers: { layer: PixelLayer; colorIdx: number }[],
  totalModels: number,
): CompositeGrid {
  // Find grid bounds
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

  // At each coordinate, collect all model contributions
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

      // Group by character — find the majority character
      const charGroups = new Map<string, number[]>();
      for (const { char, modelIdx } of contributions) {
        const arr = charGroups.get(char) || [];
        arr.push(modelIdx);
        charGroups.set(char, arr);
      }

      if (charGroups.size === 1) {
        // All contributing models agree on the same char
        const [char, indices] = [...charGroups.entries()][0];
        const rgb = compositeColors(indices, totalModels);
        pixels.set(`${row},${col}`, {
          char,
          rgb,
          agreementRatio: indices.length / totalModels,
          modelIndices: indices,
        });
      } else {
        // Disagreement: show the majority char, color = blend of disagreeing sources
        // Pick the char with most votes; ties broken alphabetically
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

        // Color: blend only the majority group, dimmed by disagreement
        const allIndices = contributions.map((c) => c.modelIdx);
        const rgb = compositeColors(bestIndices, totalModels);
        // Dim slightly to show disagreement
        const dimFactor = bestCount / contributions.length;
        pixels.set(`${row},${col}`, {
          char: bestChar,
          rgb: {
            r: Math.round(rgb.r * dimFactor),
            g: Math.round(rgb.g * dimFactor),
            b: Math.round(rgb.b * dimFactor),
          },
          agreementRatio: bestCount / totalModels,
          modelIndices: allIndices,
        });
      }
    }
  }

  return { rows: maxRow + 1, cols: maxCol + 1, pixels };
}

/** Render the composite grid to ANSI-colored terminal lines */
export function renderCompositeGrid(grid: CompositeGrid, maxWidth: number): string[] {
  const lines: string[] = [];

  for (let row = 0; row < grid.rows; row++) {
    let line = "";
    let lastRGB: RGB | null = null;

    for (let col = 0; col < Math.min(grid.cols, maxWidth); col++) {
      const px = grid.pixels.get(`${row},${col}`);
      if (!px) {
        if (lastRGB) { line += "\x1b[0m"; lastRGB = null; }
        line += " ";
      } else {
        // Only emit new color code if RGB changed
        if (!lastRGB || lastRGB.r !== px.rgb.r || lastRGB.g !== px.rgb.g || lastRGB.b !== px.rgb.b) {
          if (lastRGB) line += "\x1b[0m";
          line += `\x1b[38;2;${px.rgb.r};${px.rgb.g};${px.rgb.b}m`;
          lastRGB = px.rgb;
        }
        line += px.char;
      }
    }
    if (lastRGB) line += "\x1b[0m";
    lines.push(line);
  }

  return lines;
}

// ── Disagreement map: where pixels don't align ────────────────────────────────

interface Disagreement {
  row: number;
  col: number;
  variants: { char: string; modelIndices: number[] }[];
}

export function findDisagreements(
  layers: { layer: PixelLayer; colorIdx: number }[],
): Disagreement[] {
  const disagreements: Disagreement[] = [];
  let maxRow = 0, maxCol = 0;
  for (const { layer } of layers) {
    for (const [row, cols] of layer) {
      if (row > maxRow) maxRow = row;
      for (const col of cols.keys()) { if (col > maxCol) maxCol = col; }
    }
  }

  for (let row = 0; row <= maxRow; row++) {
    for (let col = 0; col <= maxCol; col++) {
      const charGroups = new Map<string, number[]>();
      for (const { layer, colorIdx } of layers) {
        const ch = layer.get(row)?.get(col);
        if (ch !== undefined) {
          const arr = charGroups.get(ch) || [];
          arr.push(colorIdx);
          charGroups.set(ch, arr);
        }
      }
      if (charGroups.size > 1) {
        disagreements.push({
          row, col,
          variants: [...charGroups.entries()].map(([char, modelIndices]) => ({ char, modelIndices })),
        });
      }
    }
  }

  return disagreements;
}

// ── SSE streaming fetch ───────────────────────────────────────────────────────

async function streamModelSCL(
  modelId: string,
  prompt: string,
  onToken: (token: string) => void,
  signal: AbortSignal,
): Promise<{ error?: string }> {
  try {
    const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: modelId,
        stream: true,
        messages: [
          { role: "system", content: SCL_SYSTEM },
          { role: "user", content: prompt },
        ],
        max_tokens: 400,
      }),
      signal,
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      return { error: (data as any).error?.message || `HTTP ${res.status}` };
    }

    const reader = res.body?.getReader();
    if (!reader) return { error: "No response body" };

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
          if (delta) onToken(delta);
        } catch { /* skip */ }
      }
    }

    return {};
  } catch (e: any) {
    if (e?.name === "AbortError") return { error: "aborted" };
    return { error: String(e) };
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function shortName(modelId: string): string {
  return modelId.split("/").pop()?.replace(":free", "") ?? modelId;
}

// Backward-compat flat merge for tests
export function mergeSCL(streams: ModelStream[]): {
  concepts: Map<string, SCLConcept>;
  contributing: number;
} {
  const merged = new Map<string, SCLConcept>();
  let contributing = 0;
  for (const s of streams) {
    if (s.error || s.concepts.length === 0) continue;
    contributing++;
    for (const c of s.concepts) {
      const existing = merged.get(c.name);
      if (!existing) {
        merged.set(c.name, { name: c.name, props: new Map(c.props), confidenceSum: c.confidenceSum, count: 1 });
      } else {
        for (const [k, v] of c.props) {
          if (k === "uncertainty") continue;
          const cur = existing.props.get(k);
          if (!cur) existing.props.set(k, v);
          else if (cur !== v && !cur.includes(v)) existing.props.set(k, `${cur} ┊ ${v}`);
        }
        existing.confidenceSum += c.confidenceSum;
        existing.count += 1;
      }
    }
  }
  return { concepts: merged, contributing };
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIVE TUI COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export class SCLPixelComponent implements Component {
  private tui: TUI;
  private theme: any;
  private done: () => void;
  private streams: ModelStream[];
  private prompt: string;
  private abortController = new AbortController();
  private startTime = Date.now();
  private scrollOffset = 0;
  private showDisagreements = false;

  constructor(
    tui: TUI,
    theme: any,
    done: () => void,
    streams: ModelStream[],
    prompt: string,
  ) {
    this.tui = tui;
    this.theme = theme;
    this.done = done;
    this.streams = streams;
    this.prompt = prompt;
  }

  async start(): Promise<void> {
    const promises = this.streams.map((stream) =>
      streamModelSCL(
        stream.modelId,
        this.prompt,
        (token) => {
          stream.buffer += token;
          stream.tokenCount++;
          stream.concepts = parseSCLIncremental(stream.buffer).complete;
          this.tui.requestRender();
        },
        this.abortController.signal,
      ).then((result) => {
        stream.done = true;
        if (result.error) stream.error = result.error;
        stream.concepts = parseSCL(stream.buffer);
        this.tui.requestRender();
      }),
    );

    await Promise.allSettled(promises);
    this.tui.requestRender();
  }

  handleInput(data: string): void {
    if (data === "\x1b" || data === "\x03") {
      this.abortController.abort();
      this.done();
    } else if (data === "\r") {
      if (this.streams.every((s) => s.done)) this.done();
    } else if (data === "\x1b[A") {
      this.scrollOffset = Math.max(0, this.scrollOffset - 1);
      this.tui.requestRender();
    } else if (data === "\x1b[B") {
      this.scrollOffset++;
      this.tui.requestRender();
    } else if (data === "d" || data === "D") {
      this.showDisagreements = !this.showDisagreements;
      this.tui.requestRender();
    }
  }

  invalidate(): void {}

  render(width: number): string[] {
    const innerW = Math.max(30, width - 2);
    const lines: string[] = [];
    const elapsed = ((Date.now() - this.startTime) / 1000).toFixed(1);

    const doneCount = this.streams.filter((s) => s.done).length;
    const errCount = this.streams.filter((s) => s.error).length;
    const totalTokens = this.streams.reduce((a, s) => a + s.tokenCount, 0);
    const activeStreams = this.streams.filter((s) => !s.done);
    const validStreams = this.streams.filter((s) => !s.error && s.concepts.length > 0);

    // ── Header ────────────────────────────────────────────────────────────
    lines.push("");
    const icon = activeStreams.length > 0 ? "◈" : "◆";
    lines.push(
      ansiBold(` ${icon} SCL Pixel Consensus`) +
      ansiDim(` — ${doneCount}/${this.streams.length} models  ${totalTokens} tok  ${elapsed}s`),
    );

    // ── Active streams ────────────────────────────────────────────────────
    if (activeStreams.length > 0) {
      const parts = activeStreams.map((s) =>
        ansiModelColor(s.colorIdx, `${shortName(s.modelId)}(${s.tokenCount})`),
      );
      lines.push(` ⟳ ${parts.join("  ")}`);
    }
    if (errCount > 0) {
      lines.push(` \x1b[31m⚠ ${errCount} error${errCount > 1 ? "s" : ""}\x1b[0m`);
    }
    lines.push("");

    // ── Legend ─────────────────────────────────────────────────────────────
    const legendModels = this.streams.filter((s) => !s.error);
    let legendLine = " ";
    let legendVisLen = 1;
    for (const s of legendModels) {
      const name = shortName(s.modelId);
      const entryLen = name.length + 4;
      if (legendVisLen > 1 && legendVisLen + entryLen > innerW - 2) {
        lines.push(legendLine);
        legendLine = " ";
        legendVisLen = 1;
      }
      legendLine += ansiModelColor(s.colorIdx, `● ${name}`) + "  ";
      legendVisLen += entryLen;
    }
    if (legendVisLen > 1) lines.push(legendLine);

    // Consensus legend
    lines.push(
      "  " +
      ansiTrueColor({ r: 255, g: 255, b: 255 }, "█") + ansiDim("=unanimous") + "  " +
      ansiTrueColor({ r: 180, g: 180, b: 180 }, "█") + ansiDim("=strong") + "  " +
      ansiTrueColor({ r: 231, g: 76, b: 60 }, "█") + ansiDim("=minority") + "  " +
      ansiDim("·") + ansiDim("=empty"),
    );
    lines.push("");

    // ── Build pixel layers and composite ──────────────────────────────────
    if (validStreams.length > 0) {
      const layers = validStreams.map((s) => ({
        layer: textToPixelLayer(canonicalLayout(s.concepts)),
        colorIdx: s.colorIdx,
      }));

      const grid = compositeLayersToGrid(layers, this.streams.filter((s) => !s.error).length);
      const composited = renderCompositeGrid(grid, innerW - 2);

      lines.push(ansiDim(` ── composited (${validStreams.length} layers) ──`));
      lines.push("");
      for (const cl of composited) {
        lines.push(` ${cl}`);
      }
      lines.push("");

      // ── Agreement stats ─────────────────────────────────────────────────
      let totalPx = 0, unanimousPx = 0, disagreePx = 0;
      for (const px of grid.pixels.values()) {
        totalPx++;
        if (px.agreementRatio >= 1.0) unanimousPx++;
      }
      const disagreements = findDisagreements(layers);
      disagreePx = disagreements.length;
      const agreePercent = totalPx > 0 ? Math.round((unanimousPx / totalPx) * 100) : 0;

      const statsLine =
        ansiTrueColor({ r: 255, g: 255, b: 255 }, `${agreePercent}% unanimous`) +
        ansiDim(` (${unanimousPx}/${totalPx} px)  `) +
        (disagreePx > 0
          ? `\x1b[33m${disagreePx} disagreement${disagreePx > 1 ? "s" : ""}\x1b[0m`
          : ansiTrueColor({ r: 100, g: 255, b: 100 }, "0 disagreements"));
      lines.push(` ${statsLine}`);
      lines.push("");

      // ── Disagreement detail (toggle with d) ─────────────────────────────
      if (this.showDisagreements && disagreements.length > 0) {
        lines.push(ansiDim(` ── disagreements ──`));
        for (const d of disagreements.slice(0, 20)) {
          const parts = d.variants.map((v) => {
            const who = v.modelIndices.map((i) => ansiModelColor(i, shortName(this.streams[i]?.modelId ?? "?"))).join(",");
            return `${ansiModelColor(v.modelIndices[0], `'${v.char}'`)}←${who}`;
          });
          lines.push(`  [${d.row},${d.col}] ${parts.join("  vs  ")}`);
        }
        if (disagreements.length > 20) {
          lines.push(ansiDim(`  ... and ${disagreements.length - 20} more`));
        }
        lines.push("");
      }
    } else if (activeStreams.length > 0) {
      // Still streaming, no complete concepts yet — show forming
      lines.push(ansiDim(" ⋯ waiting for SCL blocks to register…"));
      lines.push("");

      const partials = activeStreams
        .map((s) => {
          const { partial } = parseSCLIncremental(s.buffer);
          if (!partial) return null;
          return { stream: s, text: partial };
        })
        .filter(Boolean) as { stream: ModelStream; text: string }[];

      if (partials.length > 0) {
        lines.push(ansiDim(` ── forming ──`));
        for (const p of partials) {
          const name = shortName(p.stream.modelId);
          const preview = p.text.replace(/\n/g, " ").trim();
          const maxLen = innerW - name.length - 6;
          const shown = preview.length > maxLen ? preview.slice(0, maxLen - 1) + "…" : preview;
          lines.push(` ${ansiModelColor(p.stream.colorIdx, name + ":")} ${ansiModelColor(p.stream.colorIdx, shown)}`);
        }
        lines.push("");
      }
    } else {
      lines.push(ansiDim(" (no SCL concepts extracted from any model)"));
      lines.push("");
    }

    // ── Per-model detail (when all done) ──────────────────────────────────
    if (activeStreams.length === 0 && doneCount > 0) {
      lines.push(ansiDim(` ── per-model layers ──`));
      for (const s of this.streams) {
        const name = shortName(s.modelId);
        if (s.error) {
          lines.push(`  \x1b[31m✗\x1b[0m ${ansiModelColor(s.colorIdx, name)}: ${ansiDim(s.error.slice(0, innerW - name.length - 8))}`);
        } else {
          const cnames = s.concepts.map((c) => `@${c.name}`).join(" ");
          lines.push(`  ${ansiModelColor(s.colorIdx, "✓ " + name)}: ${cnames || "(empty)"} ${ansiDim(`[${s.tokenCount} tok]`)}`);
        }
      }
      lines.push("");
    }

    // ── Footer ────────────────────────────────────────────────────────────
    const hints: string[] = [];
    if (activeStreams.length > 0) hints.push("Esc cancel");
    else hints.push("Enter/Esc close");
    hints.push("↑↓ scroll");
    hints.push(`d toggle disagreements${this.showDisagreements ? " [ON]" : ""}`);
    lines.push(ansiDim(` ${hints.join(" • ")}`));

    // ── Scroll ────────────────────────────────────────────────────────────
    const maxVisible = 46;
    const maxScroll = Math.max(0, lines.length - maxVisible);
    this.scrollOffset = Math.min(this.scrollOffset, maxScroll);
    const visible = lines.slice(this.scrollOffset, this.scrollOffset + maxVisible);

    return this.box(visible, width, "SCL Pixel Consensus");
  }

  private box(lines: string[], width: number, title: string): string[] {
    const innerW = Math.max(1, width - 2);
    const topPad = Math.max(0, innerW - title.length - 4);
    const topLine = `─ ${ansiBold(title)} ${"─".repeat(topPad)}`;

    const out: string[] = [];
    out.push(`╭${topLine}╮`);
    for (const line of lines) {
      const visible = line.replace(/\x1b\[[^m]*m/g, "");
      const pad = Math.max(0, innerW - visible.length);
      out.push(`│${line}${" ".repeat(pad)}│`);
    }
    out.push(`╰${"─".repeat(innerW)}╯`);
    return out;
  }
}
