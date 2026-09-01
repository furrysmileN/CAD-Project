import { useMemo, useState } from "react";
import ModelViewer from "./components/ModelViewer";
import TriViewPreview from "./components/TriViewPreview";


type ValidationIssue = {
  code: string;
  path: string;
  message: string;
  severity: string;
};

type ValidationResponse = {
  valid: boolean;
  issues: ValidationIssue[];
};

type RunResponse = {
  runId: string;
  status: string;
  validation: ValidationResponse;
  modelUrl: string | null;
  stepUrl: string | null;
  generatedCode: string | null;
  metrics: {
    validShape: boolean;
    volume: number;
    bboxSize: number[];
    bboxCenter: number[];
    canonicalFrame: boolean;
    runtimeSec: number;
  } | null;
  error: string | null;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
const DEFAULT_PLAN = JSON.stringify(
  {
    schema_version: "harnesscad.plan.v1",
    sample_id: "demo_box_with_hole",
    coordinate_system: {
      units: "normalized",
      origin: [0.0, 0.0, 0.0],
      longest_bbox_edge: 1.0
    },
    metadata: {
      description: "Normalized box with one through-hole"
    },
    operations: [
      {
        id: "base",
        primitive: "box",
        combine: "new",
        center: [0.0, 0.0, 0.0],
        size: [1.0, 0.6, 0.2]
      },
      {
        id: "center_hole",
        primitive: "cylinder",
        combine: "cut",
        center: [0.0, 0.0, 0.0],
        radius: 0.1,
        height: 0.4,
        axis: [0.0, 0.0, 1.0]
      }
    ]
  },
  null,
  2
);

export default function HarnessApp() {
  const [planText, setPlanText] = useState(DEFAULT_PLAN);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [result, setResult] = useState<RunResponse | null>(null);

  const modelUrl = useMemo(() => {
    if (!result?.modelUrl) {
      return "";
    }
    return result.modelUrl.startsWith("http") ? result.modelUrl : `${API_BASE_URL}${result.modelUrl}`;
  }, [result]);

  const parsePlan = () => {
    const parsed = JSON.parse(planText) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("Plan 根节点必须是 JSON object。");
    }
    return parsed as Record<string, unknown>;
  };

  const request = async (mode: "validate" | "run") => {
    setLoading(true);
    setError("");
    if (mode === "run") {
      setResult(null);
    }
    try {
      const plan = parsePlan();
      const response = await fetch(`${API_BASE_URL}/api/harness/${mode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mode === "run" ? { plan, timeout_sec: 30 } : { plan })
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as { detail?: string };
        throw new Error(body.detail ?? `HTTP ${response.status}`);
      }
      if (mode === "validate") {
        setValidation((await response.json()) as ValidationResponse);
      } else {
        const data = (await response.json()) as RunResponse;
        setResult(data);
        setValidation(data.validation);
        if (data.status !== "success") {
          setError(data.error ?? `运行状态：${data.status}`);
        }
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "请求失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <h1>HarnessCAD</h1>
      <p className="subtitle">受约束 CAD Plan → 静态校验 → 确定性编译 → CadQuery 子进程 → STEP/STL + episode trace</p>
      <p className="project-description">
        当前 v1 只支持 box / cylinder / sphere 和 new / add / cut / intersect。所有坐标均为 normalized unit-bbox。
      </p>

      <div className="panel">
        <h2>CAD Plan JSON</h2>
        <textarea
          value={planText}
          onChange={(event) => setPlanText(event.target.value)}
          rows={28}
          spellCheck={false}
          style={{ width: "100%", fontFamily: "Consolas, monospace", fontSize: "13px" }}
        />
        <div className="code-toolbar">
          <button onClick={() => void request("validate")} disabled={loading}>
            {loading ? "处理中..." : "只校验 Plan"}
          </button>
          <button onClick={() => void request("run")} disabled={loading}>
            {loading ? "执行中..." : "校验并生成 CAD"}
          </button>
          <a href="/">返回原 Demo</a>
        </div>
        {validation && (
          <p className={validation.valid ? "matched" : "error"}>
            Plan validation：{validation.valid ? "PASS" : `FAIL (${validation.issues.length})`}
          </p>
        )}
        {validation && validation.issues.length > 0 && <pre>{JSON.stringify(validation.issues, null, 2)}</pre>}
        {error && <p className="error">{error}</p>}
      </div>

      {result?.status === "success" && result.metrics && (
        <>
          <div className="panel">
            <h2>Episode 结果</h2>
            <p className="matched">
              run_id：<code>{result.runId}</code>；shape valid：<code>{String(result.metrics.validShape)}</code>；
              canonical frame：<code>{String(result.metrics.canonicalFrame)}</code>
            </p>
            <p className="hint">
              bbox = {result.metrics.bboxSize.map((value) => value.toFixed(4)).join(" × ")}；
              volume = {result.metrics.volume.toFixed(8)}；runtime = {result.metrics.runtimeSec.toFixed(3)} s
            </p>
            {result.stepUrl && (
              <a href={`${API_BASE_URL}${result.stepUrl}`} download>
                下载 result.step
              </a>
            )}
          </div>

          <div className="grid">
            <div className="panel">
              <h2>3D Model Viewer</h2>
              <ModelViewer
                modelUrl={modelUrl}
                renderMode="solid_edges"
                showDimensions={false}
                preciseMeasure={false}
              />
            </div>
            <div className="panel">
              <h2>确定性生成代码</h2>
              <pre>{result.generatedCode}</pre>
            </div>
          </div>

          <div className="panel">
            <TriViewPreview modelUrl={modelUrl} />
          </div>
        </>
      )}
    </div>
  );
}
