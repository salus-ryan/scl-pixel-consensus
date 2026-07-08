import { useState, useRef, useCallback, useEffect } from "react";
import { PixelCanvas } from "./PixelCanvas";
import { parseSCL, canonicalLayout, textToPixelLayer, compositeGrid, modelColor, MODEL_COLORS, type RGB, type CompositePixel } from "./scl";

interface ModelState {
  id: string;
  name: string;
  colorIdx: number;
  buffer: string;
  tokens: number;
  done: boolean;
  error?: string;
  retrying?: number;
}

export default function App() {
  const [prompt, setPrompt] = useState("");
  const [running, setRunning] = useState(false);
  const [models, setModels] = useState<ModelState[]>([]);
  const [grid, setGrid] = useState<{ rows: number; cols: number; pixels: Map<string, CompositePixel> } | null>(null);
  const [stats, setStats] = useState({ consensus: 0, strong: 0, unanimous: 0, total: 0, disagreements: 0 });
  const abortRef = useRef<AbortController | null>(null);
  const modelsRef = useRef<ModelState[]>([]);

  const recomposite = useCallback((currentModels: ModelState[]) => {
    const validLayers = currentModels
      .filter((m) => !m.error && m.buffer.length > 0)
      .map((m) => {
        const concepts = parseSCL(m.buffer);
        if (concepts.length === 0) return null;
        const layout = canonicalLayout(concepts);
        const layer = textToPixelLayer(layout);
        return { layer, colorIdx: m.colorIdx };
      })
      .filter(Boolean) as { layer: any; colorIdx: number }[];

    if (validLayers.length === 0) {
      setGrid(null);
      setStats({ consensus: 0, strong: 0, unanimous: 0, total: 0, disagreements: 0 });
      return;
    }

    const totalModels = currentModels.filter((m) => !m.error).length;
    const g = compositeGrid(validLayers, totalModels);
    setGrid(g);

    let consensus = 0; // 2+ models agree on same char at same pixel
    let strong = 0;     // majority of models agree
    let unanimous = 0;  // all models agree
    let disagreements = 0;
    for (const px of g.pixels.values()) {
      if (px.modelIndices.length >= 2) consensus++;
      if (px.agreementRatio > 0.5) strong++;
      if (px.agreementRatio >= 1.0) unanimous++;
      if (px.modelIndices.length > 1 && px.agreementRatio < 1.0) disagreements++;
    }
    setStats({ consensus, strong, unanimous, total: g.pixels.size, disagreements });
  }, []);

  const startStream = useCallback(async () => {
    if (!prompt.trim()) return;
    setRunning(true);
    setGrid(null);
    setModels([]);
    setStats({ consensus: 0, strong: 0, unanimous: 0, total: 0, disagreements: 0 });

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch(`/api/stream?prompt=${encodeURIComponent(prompt)}`, {
        signal: controller.signal,
      });

      const reader = res.body?.getReader();
      if (!reader) return;

      const decoder = new TextDecoder();
      let pending = "";
      let tokenBatch = 0;

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

          try {
            const msg = JSON.parse(payload);

            if (msg.type === "models") {
              const initial: ModelState[] = msg.models.map((m: any, i: number) => ({
                id: m.id,
                name: m.name || m.id.split("/").pop()?.replace(":free", "") || m.id,
                colorIdx: i,
                buffer: "",
                tokens: 0,
                done: false,
              }));
              modelsRef.current = initial;
              setModels([...initial]);
            } else if (msg.type === "token") {
              const m = modelsRef.current[msg.idx];
              if (m) {
                m.buffer += msg.token;
                m.tokens++;
                tokenBatch++;
                // Recomposite every 5 tokens to avoid thrashing
                if (tokenBatch >= 5) {
                  tokenBatch = 0;
                  setModels([...modelsRef.current]);
                  recomposite(modelsRef.current);
                }
              }
            } else if (msg.type === "model_retry") {
              const m = modelsRef.current[msg.idx];
              if (m) {
                m.retrying = msg.attempt;
                setModels([...modelsRef.current]);
              }
            } else if (msg.type === "model_done") {
              const m = modelsRef.current[msg.idx];
              if (m) {
                m.done = true;
                m.retrying = undefined;
                setModels([...modelsRef.current]);
                recomposite(modelsRef.current);
              }
            } else if (msg.type === "model_error") {
              const m = modelsRef.current[msg.idx];
              if (m) {
                m.done = true;
                m.error = msg.error;
                setModels([...modelsRef.current]);
              }
            } else if (msg.type === "done") {
              // Final recomposite
              setModels([...modelsRef.current]);
              recomposite(modelsRef.current);
            }
          } catch { /* skip */ }
        }
      }
    } catch (e: any) {
      if (e?.name !== "AbortError") {
        console.error("Stream error:", e);
      }
    }

    setRunning(false);
  }, [prompt, recomposite]);

  const cancel = () => {
    abortRef.current?.abort();
    setRunning(false);
  };

  const doneCount = models.filter((m) => m.done).length;
  const errorCount = models.filter((m) => m.error).length;
  const totalTokens = models.reduce((a, m) => a + m.tokens, 0);
  const consensusPercent = stats.total > 0 ? Math.round((stats.consensus / stats.total) * 100) : 0;
  const strongPercent = stats.total > 0 ? Math.round((stats.strong / stats.total) * 100) : 0;
  const unanimousPercent = stats.total > 0 ? Math.round((stats.unanimous / stats.total) * 100) : 0;

  return (
    <div className="min-h-screen flex flex-col p-4 gap-4 max-w-[1400px] mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-bold tracking-tight">
          <span className="text-white">◈</span> SCL Pixel Consensus
        </h1>
        <span className="text-gray-500 text-sm">visual ensemble via character-level registration</span>
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !running && startStream()}
          placeholder="Enter a prompt to compress across models..."
          className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 transition-colors"
          disabled={running}
        />
        {running ? (
          <button onClick={cancel} className="px-5 py-2 bg-red-600 hover:bg-red-700 rounded-lg font-medium transition-colors">
            Cancel
          </button>
        ) : (
          <button onClick={startStream} className="px-5 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium transition-colors disabled:opacity-50" disabled={!prompt.trim()}>
            Stream
          </button>
        )}
      </div>

      {/* Stats bar */}
      {models.length > 0 && (
        <div className="flex items-center gap-4 text-sm text-gray-400">
          <span>{doneCount}/{models.length} models</span>
          <span>{totalTokens} tokens</span>
          {stats.total > 0 && (
            <>
              <span className="text-green-400 font-medium">{consensusPercent}% overlap</span>
              <span className="text-gray-500">({stats.consensus} px with 2+ models)</span>
              <span className="text-blue-300">{strongPercent}% strong</span>
              {stats.unanimous > 0 && <span className="text-white">{unanimousPercent}% unanimous</span>}
              {stats.disagreements > 0 && (
                <span className="text-yellow-400">{stats.disagreements} conflicts</span>
              )}
            </>
          )}
          {errorCount > 0 && <span className="text-red-400">{errorCount} errors</span>}
        </div>
      )}

      {/* Main content */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Pixel grid */}
        <div className="flex-1 bg-gray-900/50 border border-gray-800 rounded-xl p-4 overflow-auto">
          {grid ? (
            <PixelCanvas grid={grid} />
          ) : running ? (
            <div className="flex items-center justify-center h-64 text-gray-500">
              <div className="text-center">
                <div className="text-2xl mb-2 animate-pulse">◈</div>
                <div>Waiting for SCL blocks to register...</div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-600">
              Enter a prompt and press Stream to begin
            </div>
          )}
        </div>

        {/* Model sidebar */}
        {models.length > 0 && (
          <div className="w-64 bg-gray-900/50 border border-gray-800 rounded-xl p-3 overflow-y-auto flex flex-col gap-1">
            <div className="text-xs text-gray-500 font-medium uppercase tracking-wider mb-2">Models</div>
            {models.map((m) => {
              const c = modelColor(m.colorIdx);
              const shortName = m.id.split("/").pop()?.replace(":free", "") || m.id;
              return (
                <div key={m.id} className="flex items-center gap-2 text-xs py-1">
                  <div
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: `rgb(${c.r},${c.g},${c.b})` }}
                  />
                  <span className={`truncate ${m.error ? "text-red-400 line-through" : m.retrying ? "text-yellow-400" : m.done ? "text-gray-300" : "text-gray-500"}`}>
                    {shortName}
                  </span>
                  {m.retrying && <span className="text-yellow-500 text-[10px]">↻{m.retrying}</span>}
                  <span className="ml-auto text-gray-600">{m.tokens}</span>
                  {!m.done && !m.error && !m.retrying && (
                    <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                  )}
                  {m.retrying && (
                    <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 animate-pulse" />
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Legend */}
      {grid && (
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-white inline-block" /> unanimous
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-gray-400 inline-block" /> strong agreement
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded inline-block" style={{ backgroundColor: `rgb(231,76,60)` }} /> minority
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-gray-800 inline-block" /> empty
          </span>
        </div>
      )}
    </div>
  );
}
