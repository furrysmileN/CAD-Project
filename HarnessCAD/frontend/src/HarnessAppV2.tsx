import { useMemo, useState } from "react";
import ModelViewer from "./components/ModelViewer";
import TriViewPreview from "./components/TriViewPreview";


type Issue = { code: string; path?: string; message: string; severity: string; operationIndex?: number | null; operationId?: string | null };
type ShapeMetrics = {
  shapeType: string | null;
  valid: boolean;
  solidCount: number;
  shellCount: number;
  faceCount: number;
  edgeCount: number;
  vertexCount: number;
  volume: number;
  area: number;
  bboxSize: number[] | null;
  bboxCenter: number[] | null;
  canonicalFrame?: boolean;
};
type OperationTrace = {
  index: number;
  id: string;
  primitive: string;
  combine: string;
  status: string;
  durationSec: number;
  before: ShapeMetrics;
  after: ShapeMetrics | null;
  volumeDelta: number | null;
  warnings: string[];
};
type StageTrace = { stage: string; status: string; durationSec: number; returncode?: number; bytes?: number };
type Failure = {
  code: string;
  stage: string;
  message: string;
  operationIndex: number | null;
  operationId: string | null;
  exceptionType?: string | null;
};
type Validation = {
  valid: boolean;
  issues: Issue[];
  warnings: Issue[];
  planSummary: {
    operationCount: number;
    primitiveCounts: Record<string, number>;
    combineCounts: Record<string, number>;
    warningCount: number;
  };
};
type Episode = {
  traceVersion: string;
  runId: string;
  status: string;
  failure: Failure | null;
  warnings: Issue[];
  validation: Validation;
  operationTrace: OperationTrace[];
  stageTrace: StageTrace[];
  metrics: ShapeMetrics | null;
  environment: Record<string, string>;
  provenance: { planSha256: string; generatedCodeSha256: string | null };
  totalDurationSec: number;
  modelUrl: string | null;
  stepUrl: string | null;
  generatedCode: string | null;
  artifactManifest: Array<{ name: string; bytes: number; sha256: string }>;
  error: string | null;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_PLAN = JSON.stringify(
  {
    schema_version: "harnesscad.plan.v1",
    sample_id: "episode_v2_box_hole",
    coordinate_system: { units: "normalized", origin: [0, 0, 0], longest_bbox_edge: 1.0 },
    metadata: { description: "Edit radius to 1.8 to trigger empty_after_operation" },
    operations: [
      { id: "base", primitive: "box", combine: "new", center: [0, 0, 0], size: [1.0, 0.6, 0.2] },
      { id: "hole", primitive: "cylinder", combine: "cut", center: [0, 0, 0], radius: 0.1, height: 0.4, axis: [0, 0, 1] }
    ]
  },
  null,
  2
);

const fmt = (value: number | null | undefined, digits = 6) =>
  value === null || value === undefined ? "—" : value.toFixed(digits);

export default function HarnessAppV2() {
  const [planText, setPlanText] = useState(DEFAULT_PLAN);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [validation, setValidation] = useState<Validation | null>(null);
  const [episode, setEpisode] = useState<Episode | null>(null);

  const modelUrl = useMemo(() => {
    if (!episode?.modelUrl) return "";
    return episode.modelUrl.startsWith("http") ? episode.modelUrl : `${API_BASE_URL}${episode.modelUrl}`;
  }, [episode]);

  const parsePlan = () => {
    const parsed = JSON.parse(planText) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("Plan 根节点必须是 JSON object。");
    return parsed as Record<string, unknown>;
  };

  const request = async (mode: "validate" | "run") => {
    setLoading(true);
    setError("");
    if (mode === "run") setEpisode(null);
    try {
      const plan = parsePlan();
      const response = await fetch(`${API_BASE_URL}/api/harness-v2/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mode === "run" ? { plan, timeout_sec: 30 } : { plan })
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${await response.text()}`);
      if (mode === "validate") {
        setValidation((await response.json()) as Validation);
      } else {
        const data = (await response.json()) as Episode;
        setEpisode(data);
        setValidation(data.validation);
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "请求失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>HarnessCAD — Episode v2</h1>
      <p className="subtitle">Plan 校验、preflight、逐操作几何状态、导出与后条件全部分别记录。</p>
      <div className="panel">
        <h2>CAD Plan JSON</h2>
        <textarea value={planText} onChange={(event) => setPlanText(event.target.value)} rows={25} spellCheck={false}
          style={{ width: "100%", fontFamily: "Consolas, monospace", fontSize: "13px" }} />
        <div className="code-toolbar">
          <button disabled={loading} onClick={() => void request("validate")}>{loading ? "处理中..." : "校验 + Preflight"}</button>
          <button disabled={loading} onClick={() => void request("run")}>{loading ? "执行中..." : "执行并记录 Episode"}</button>
          <a href="/harness.html">返回 v1</a><a href="/">原 Demo</a>
        </div>
        {error && <p className="error">{error}</p>}
      </div>

      {validation && (
        <div className="panel">
          <h2>1. Plan Validation / Preflight</h2>
          <p className={validation.valid ? "matched" : "error"}>Schema + semantic：{validation.valid ? "PASS" : "FAIL"}</p>
          <p className="hint">operations={validation.planSummary.operationCount}；primitives={JSON.stringify(validation.planSummary.primitiveCounts)}；booleans={JSON.stringify(validation.planSummary.combineCounts)}</p>
          {validation.issues.map((item, index) => <p className="error" key={`issue-${index}`}><code>{item.code}</code> {item.path}：{item.message}</p>)}
          {validation.warnings.map((item, index) => <p className="hint" key={`warning-${index}`}><code>{item.code}</code> op={item.operationId ?? "—"}：{item.message}</p>)}
        </div>
      )}

      {episode && (
        <>
          <div className="panel">
            <h2>2. Episode Summary</h2>
            <p className={episode.status.startsWith("success") ? "matched" : "error"}>status=<code>{episode.status}</code>；run_id=<code>{episode.runId}</code>；trace=<code>{episode.traceVersion}</code></p>
            <p className="hint">total={fmt(episode.totalDurationSec, 3)} s；plan SHA256=<code>{episode.provenance.planSha256.slice(0, 16)}…</code></p>
            {episode.failure && <div className="error"><strong>{episode.failure.code}</strong> — stage={episode.failure.stage}，operation={episode.failure.operationId ?? "—"}：{episode.failure.message}</div>}
            {episode.warnings.map((item, index) => <p className="hint" key={`episode-warning-${index}`}><code>{item.code}</code>：{item.message}</p>)}
          </div>

          <div className="panel">
            <h2>3. Stage Trace</h2>
            <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th>stage</th><th>status</th><th>duration</th><th>return/bytes</th></tr></thead>
              <tbody>{episode.stageTrace.map((stage, index) => <tr key={`${stage.stage}-${index}`}><td>{stage.stage}</td><td>{stage.status}</td><td>{fmt(stage.durationSec, 4)} s</td><td>{stage.returncode ?? stage.bytes ?? "—"}</td></tr>)}</tbody>
            </table>
          </div>

          <div className="panel">
            <h2>4. Operation Trace</h2>
            <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "980px" }}>
              <thead><tr><th># / id</th><th>primitive</th><th>combine</th><th>status</th><th>solids</th><th>volume before → after</th><th>Δ volume</th><th>faces</th><th>duration</th><th>warnings</th></tr></thead>
              <tbody>{episode.operationTrace.map((operation) => <tr key={`${operation.index}-${operation.id}`}>
                <td>{operation.index} / {operation.id}</td><td>{operation.primitive}</td><td>{operation.combine}</td><td>{operation.status}</td>
                <td>{operation.before.solidCount} → {operation.after?.solidCount ?? "—"}</td>
                <td>{fmt(operation.before.volume)} → {fmt(operation.after?.volume)}</td><td>{fmt(operation.volumeDelta)}</td>
                <td>{operation.after?.faceCount ?? "—"}</td><td>{fmt(operation.durationSec, 4)} s</td><td>{operation.warnings.join(", ") || "—"}</td>
              </tr>)}</tbody>
            </table></div>
          </div>

          {episode.metrics && <div className="panel"><h2>5. Final Geometry</h2>
            <p className="matched">valid={String(episode.metrics.valid)}；solids={episode.metrics.solidCount}；faces={episode.metrics.faceCount}；canonical={String(episode.metrics.canonicalFrame)}</p>
            <p className="hint">bbox={episode.metrics.bboxSize?.map((value) => value.toFixed(5)).join(" × ") ?? "—"}；volume={fmt(episode.metrics.volume, 8)}；area={fmt(episode.metrics.area, 8)}</p>
          </div>}

          {modelUrl && <><div className="grid"><div className="panel"><h2>6. 3D Result</h2><ModelViewer modelUrl={modelUrl} renderMode="solid_edges" showDimensions={false} preciseMeasure={false} /></div>
            <div className="panel"><h2>Deterministic Code</h2><pre>{episode.generatedCode}</pre></div></div>
            <div className="panel"><TriViewPreview modelUrl={modelUrl} /></div></>}

          <div className="panel"><h2>7. Artifact Manifest</h2><pre>{JSON.stringify(episode.artifactManifest, null, 2)}</pre></div>
        </>
      )}
    </div>
  );
}
