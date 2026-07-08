import { useState, useRef, useCallback } from "react";
import { SemanticCanvas, type SemanticGridData } from "./SemanticCanvas";
import { modelColor } from "./scl";

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
  const [semanticGrid, setSemanticGrid] = useState<SemanticGridData | null>(null);
  const [stats, setStats] = useState({ clusters: 0, consensus: 0, branches: 0, total: 0 });
  const [pipelineStatus, setPipelineStatus] = useState<"idle" | "running" | "done">("idle");
  const abortRef = useRef<AbortController | null>(null);
  const modelsRef = useRef<ModelState[]>([]);
  const semanticDebounce = useRef<number | null>(null);

  const runSemantic = useCallback(async (currentModels: ModelState[]) => {
    const validModels = currentModels.filter((m) => !m.error && m.buffer.length > 0);
    if (validModels.length < 1) return;

    setPipelineStatus("running");
    try {
      const resp = await fetch("/api/semantic", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          models: currentModels.map((m) => ({
            modelIdx: m.colorIdx,
            buffer: m.buffer,
            error: m.error,
          })),
        }),
      });
      if (resp.ok) {
        const grid = await resp.json() as SemanticGridData;
        setSemanticGrid(grid);

        // Compute stats from semantic grid
        let consensus = 0, branches = 0;
        for (const px of grid.pixels) {
          if (px.modelIndices.length >= 2) consensus++;
          if (px.isBranch) branches++;
        }
        setStats({
          clusters: grid.clusters.length,
          consensus,
          branches,
          total: grid.pixels.length,
        });
      }
    } catch (e) {
      console.error("Semantic pipeline error:", e);
    }
    setPipelineStatus("done");
  }, []);

  const debouncedSemantic = useCallback((currentModels: ModelState[]) => {
    if (semanticDebounce.current) clearTimeout(semanticDebounce.current);
    semanticDebounce.current = window.setTimeout(() => runSemantic(currentModels), 800);
  }, [runSemantic]);

  const startStream = useCallback(async () => {
    if (!prompt.trim()) return;
    setRunning(true);
    setSemanticGrid(null);
    setModels([]);
    setStats({ clusters: 0, consensus: 0, branches: 0, total: 0 });
    setPipelineStatus("idle");

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
                if (tokenBatch >= 20) {
                  tokenBatch = 0;
                  setModels([...modelsRef.current]);
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
                debouncedSemantic(modelsRef.current);
              }
            } else if (msg.type === "model_error") {
              const m = modelsRef.current[msg.idx];
              if (m) {
                m.done = true;
                m.error = msg.error;
                setModels([...modelsRef.current]);
              }
            } else if (msg.type === "done") {
              setModels([...modelsRef.current]);
              // Final semantic pass
              runSemantic(modelsRef.current);
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
  }, [prompt, runSemantic, debouncedSemantic]);

  const cancel = () => {
    abortRef.current?.abort();
    setRunning(false);
  };

  const doneCount = models.filter((m) => m.done).length;
  const errorCount = models.filter((m) => m.error).length;
  const totalTokens = models.reduce((a, m) => a + m.tokens, 0);
  const totalModels = models.filter((m) => !m.error).length;
  const consensusPercent = stats.total > 0 ? Math.round((stats.consensus / stats.total) * 100) : 0;

  return (
    <div className="min-h-screen flex flex-col p-4 gap-4 max-w-[1400px] mx-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-bold tracking-tight">
          <span className="text-white">◈</span> SCL Pixel Consensus
        </h1>
        <span className="text-gray-500 text-sm">semantic interference via concept-level registration</span>
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
              <span className="text-purple-400">{stats.clusters} clusters</span>
              <span className="text-green-400 font-medium">{consensusPercent}% consensus</span>
              <span className="text-gray-500">({stats.consensus} px overlap)</span>
              {stats.branches > 0 && (
                <span className="text-yellow-400">{stats.branches} branches</span>
              )}
            </>
          )}
          {pipelineStatus === "running" && <span className="text-blue-400 animate-pulse">⟳ semantic...</span>}
          {errorCount > 0 && <span className="text-red-400">{errorCount} errors</span>}
        </div>
      )}

      {/* Main content */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Semantic pixel grid */}
        <div className="flex-1 bg-gray-900/50 border border-gray-800 rounded-xl p-4 overflow-auto">
          {semanticGrid && semanticGrid.pixels.length > 0 ? (
            <SemanticCanvas grid={semanticGrid} totalModels={totalModels} />
          ) : running ? (
            <div className="flex items-center justify-center h-64 text-gray-500">
              <div className="text-center">
                <div className="text-2xl mb-2 animate-pulse">◈</div>
                <div>Streaming SCL... semantic registration pending</div>
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
      {semanticGrid && semanticGrid.pixels.length > 0 && (
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-white inline-block" /> full consensus
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-blue-300 inline-block" /> partial overlap
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded inline-block border-l-2 border-yellow-400 bg-yellow-400/20" /> value branch
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded bg-gray-700 inline-block" /> single model
          </span>
          <span className="text-gray-600 ml-2">≡ exact &nbsp; ≈ alias &nbsp; ~ embedding</span>
        </div>
      )}
    </div>
  );
}
