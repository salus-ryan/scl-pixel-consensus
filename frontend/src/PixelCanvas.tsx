import { useRef, useEffect, useMemo } from "react";
import type { CompositePixel } from "./scl";

interface Props {
  grid: { rows: number; cols: number; pixels: Map<string, CompositePixel> };
}

const CELL_SIZE = 10; // px per character cell
const FONT_SIZE = 9;

export function PixelCanvas({ grid }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { width, height } = useMemo(() => ({
    width: grid.cols * CELL_SIZE,
    height: grid.rows * CELL_SIZE,
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
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, width, height);

    // Draw grid background dots
    ctx.fillStyle = "#1a1a22";
    for (let row = 0; row < grid.rows; row++) {
      for (let col = 0; col < grid.cols; col++) {
        const x = col * CELL_SIZE;
        const y = row * CELL_SIZE;
        ctx.fillRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);
      }
    }

    // Draw pixels — overlap = brighter + glow
    ctx.font = `bold ${FONT_SIZE}px "JetBrains Mono", "Fira Code", "SF Mono", monospace`;
    ctx.textBaseline = "top";

    for (const [key, px] of grid.pixels) {
      const [rowStr, colStr] = key.split(",");
      const row = parseInt(rowStr);
      const col = parseInt(colStr);
      const x = col * CELL_SIZE;
      const y = row * CELL_SIZE;

      const modelCount = px.modelIndices.length;
      const isOverlap = modelCount >= 2;
      const agreement = px.agreementRatio;

      if (isOverlap) {
        // GLOW: outer soft glow for overlapping pixels
        const glowRadius = 2 + agreement * 4;
        const gradient = ctx.createRadialGradient(
          x + CELL_SIZE / 2, y + CELL_SIZE / 2, 0,
          x + CELL_SIZE / 2, y + CELL_SIZE / 2, glowRadius + CELL_SIZE / 2,
        );
        // Brighter glow = more agreement. White for unanimous, colored for partial.
        const glowR = agreement >= 1.0 ? 255 : Math.min(255, px.rgb.r + 100);
        const glowG = agreement >= 1.0 ? 255 : Math.min(255, px.rgb.g + 100);
        const glowB = agreement >= 1.0 ? 255 : Math.min(255, px.rgb.b + 100);
        gradient.addColorStop(0, `rgba(${glowR},${glowG},${glowB},${0.4 + agreement * 0.5})`);
        gradient.addColorStop(1, `rgba(${glowR},${glowG},${glowB},0)`);
        ctx.fillStyle = gradient;
        ctx.fillRect(x - 3, y - 3, CELL_SIZE + 6, CELL_SIZE + 6);

        // Solid bright background
        const bgAlpha = 0.5 + agreement * 0.4;
        ctx.fillStyle = `rgba(${glowR},${glowG},${glowB},${bgAlpha})`;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        // Character in white/bright
        ctx.fillStyle = agreement >= 1.0 ? "#ffffff" : `rgb(${glowR},${glowG},${glowB})`;
        ctx.fillText(px.char, x + 1, y + 1);
      } else {
        // Single model: dim, subtle
        const alpha = 0.25;
        ctx.fillStyle = `rgba(${px.rgb.r},${px.rgb.g},${px.rgb.b},${alpha})`;
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

        // Character at reduced opacity
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = `rgb(${px.rgb.r},${px.rgb.g},${px.rgb.b})`;
        ctx.fillText(px.char, x + 1, y + 1);
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
    </div>
  );
}
