import { useRef, useEffect, useMemo } from "react";

interface RGB { r: number; g: number; b: number }

export interface SemanticPixel {
  row: number;
  col: number;
  char: string;
  rgb: RGB;
  modelIndices: number[];
  agreementRatio: number;
  clusterId: string;
  assertionKey?: string;
  assertionValue?: string;
  isBranch: boolean;
}

export interface ClusterInfo {
  id: string;
  conceptNames: string[];
  normalizedName: string;
  matchType: "exact" | "alias" | "embedding";
  similarity: number;
  modelIndices: number[];
  assertions: {
    key: string;
    values: { value: string; modelIndices: number[] }[];
  }[];
  confidence: number;
}

export interface SemanticGridData {
  rows: number;
  cols: number;
  pixels: SemanticPixel[];
  clusters: ClusterInfo[];
}

interface Props {
  grid: SemanticGridData;
  totalModels: number;
}

const CELL_SIZE = 12;
const FONT_SIZE = 10;

export function SemanticCanvas({ grid, totalModels }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { width, height } = useMemo(() => ({
    width: Math.max(400, grid.cols * CELL_SIZE + 20),
    height: Math.max(100, grid.rows * CELL_SIZE + 20),
  }), [grid.cols, grid.rows]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Clear
    ctx.fillStyle = "#08080c";
    ctx.fillRect(0, 0, width, height);

    // Draw semantic pixels
    ctx.font = `bold ${FONT_SIZE}px "JetBrains Mono", "Fira Code", "SF Mono", monospace`;
    ctx.textBaseline = "top";

    for (const px of grid.pixels) {
      const x = px.col * CELL_SIZE + 10;
      const y = px.row * CELL_SIZE + 10;

      const modelCount = px.modelIndices.length;
      const isOverlap = modelCount >= 2;
      const agreement = px.agreementRatio;

      if (isOverlap) {
        // GLOW for consensus pixels
        const glowRadius = 3 + agreement * 5;
        const gradient = ctx.createRadialGradient(
          x + CELL_SIZE / 2, y + CELL_SIZE / 2, 0,
          x + CELL_SIZE / 2, y + CELL_SIZE / 2, glowRadius + CELL_SIZE / 2,
        );

        const glowR = agreement >= 1.0 ? 255 : Math.min(255, px.rgb.r + 80);
        const glowG = agreement >= 1.0 ? 255 : Math.min(255, px.rgb.g + 80);
        const glowB = agreement >= 1.0 ? 255 : Math.min(255, px.rgb.b + 80);
        gradient.addColorStop(0, `rgba(${glowR},${glowG},${glowB},${0.5 + agreement * 0.4})`);
        gradient.addColorStop(1, `rgba(${glowR},${glowG},${glowB},0)`);
        ctx.fillStyle = gradient;
        ctx.fillRect(x - 4, y - 4, CELL_SIZE + 8, CELL_SIZE + 8);

        // Solid background
        const bgAlpha = 0.6 + agreement * 0.3;
        ctx.fillStyle = `rgba(${glowR},${glowG},${glowB},${bgAlpha})`;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        // Character
        ctx.fillStyle = agreement >= 1.0 ? "#ffffff" : `rgb(${glowR},${glowG},${glowB})`;
        ctx.fillText(px.char, x + 2, y + 1);
      } else if (px.isBranch) {
        // Branch pixel — divergent value, show in model color but marked
        ctx.fillStyle = `rgba(${px.rgb.r},${px.rgb.g},${px.rgb.b},0.3)`;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        // Left border to indicate branch
        ctx.fillStyle = `rgba(${px.rgb.r},${px.rgb.g},${px.rgb.b},0.7)`;
        ctx.fillRect(x, y, 2, CELL_SIZE);

        ctx.globalAlpha = 0.7;
        ctx.fillStyle = `rgb(${px.rgb.r},${px.rgb.g},${px.rgb.b})`;
        ctx.fillText(px.char, x + 2, y + 1);
        ctx.globalAlpha = 1.0;
      } else {
        // Single model, no overlap
        ctx.fillStyle = `rgba(${px.rgb.r},${px.rgb.g},${px.rgb.b},0.15)`;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        ctx.globalAlpha = 0.45;
        ctx.fillStyle = `rgb(${px.rgb.r},${px.rgb.g},${px.rgb.b})`;
        ctx.fillText(px.char, x + 2, y + 1);
        ctx.globalAlpha = 1.0;
      }
    }
  }, [grid, width, height]);

  return (
    <div className="overflow-auto">
      <canvas
        ref={canvasRef}
        style={{ width: `${width}px`, height: `${height}px` }}
        className="block"
      />
      {/* Cluster legend below canvas */}
      {grid.clusters.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-3 text-xs">
          {grid.clusters.map((c) => (
            <div key={c.id} className="flex items-center gap-1.5 px-2 py-1 bg-gray-800/50 rounded">
              <span className="text-gray-400">
                {c.matchType === "exact" ? "≡" : c.matchType === "alias" ? "≈" : "~"}
              </span>
              <span className="text-gray-200 font-medium">@{c.normalizedName}</span>
              <span className="text-gray-500">
                {c.modelIndices.length}/{totalModels}
              </span>
              {c.conceptNames.length > 1 && (
                <span className="text-gray-600 text-[10px]">
                  ({c.conceptNames.join(", ")})
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
