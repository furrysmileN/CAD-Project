import { useEffect, useMemo, useState } from "react";
import DualCadViewer from "./components/DualCadViewer";
import PointCloudViewer from "./components/PointCloudViewer";
import HarnessAppV2 from "./HarnessAppV2";

type CaseListItem = {
  stem: string;
  family: string;
  difficulty: string;
  joint_quality: number | null;
  featured: boolean;
  label: string;
};

type CaseDetail = {
  stem: string;
  family: string;
  condition: string;
  condition_note: string;
  sent_modalities: string[];
  status: string;
  distance: {
    joint_quality: number | null;
    shape_only_cd: number | null;
    common_frame_cd: number | null;
    voxel_iou: number | null;
    label: string;
    gt_size: number[] | null;
    pred_size: number[] | null;
  };
  inputs: {
    images_sent: boolean;
    point_geom_sent: boolean;
    text_sent: boolean;
    image_views: string[];
    pointcloud_views: string[];
    l1: string;
    l3: string;
    point_evidence: {
      bbox_size?: number[];
      point_count?: number;
      sections?: Record<string, { bbox_size?: number[]; point_count?: number }>;
      symmetry?: Array<{ type: string; confidence: number }>;
    };
    local_guidance?: {
      thin_axis: string;
      long_axis: string;
      workplane: string;
      extent_class: string;
      thin_ratio: number;
      bbox_size: number[];
      generator: string | null;
      generator_verbs: string[];
      topology: string;
      inner_radius_known: boolean;
      inner_radius?: number | null;
      outer_radius?: number | null;
      loft_if_section_scales_differ: boolean;
    } | null;
  };
  assets: {
    gt_stl: string;
    pred_stl: string | null;
    pointcloud: string;
    images: string[];
    pointcloud_images: string[];
  };
};

const IMAGE_LABELS = ["前", "侧", "顶", "等轴"];

const fmt = (value: number | null | undefined, digits = 3) =>
  value === null || value === undefined || Number.isNaN(value) ? "—" : value.toFixed(digits);

const sizeText = (size: number[] | null | undefined) =>
  size && size.length === 3 ? size.map((item) => item.toFixed(2)).join(" × ") : "—";

export default function CompareApp() {
  const debug = new URLSearchParams(window.location.search).get("debug") === "1";
  if (debug) {
    return <HarnessAppV2 />;
  }

  const [cases, setCases] = useState<CaseListItem[]>([]);
  const [stem, setStem] = useState("i_beam_000097_s20260505");
  const [condition, setCondition] = useState("I1P_geom");
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [points, setPoints] = useState<number[][]>([]);
  const [mode, setMode] = useState<"side" | "overlay">("side");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    void fetch(`/api/compare/cases?condition=${encodeURIComponent(condition)}`)
      .then((response) => response.json())
      .then((payload) => setCases(payload.cases ?? []))
      .catch((caught: Error) => setError(caught.message));
  }, [condition]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    setPoints([]);
    void fetch(`/api/compare/cases/${encodeURIComponent(stem)}?condition=${encodeURIComponent(condition)}`)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(await response.text());
        }
        return response.json() as Promise<CaseDetail>;
      })
      .then(async (payload) => {
        if (cancelled) {
          return;
        }
        setDetail(payload);
        const cloud = await fetch(payload.assets.pointcloud);
        if (cloud.ok) {
          const body = (await cloud.json()) as { points: number[][] };
          if (!cancelled) {
            setPoints(body.points ?? []);
          }
        }
      })
      .catch((caught: Error) => {
        if (!cancelled) {
          setError(caught.message);
          setDetail(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [stem, condition]);

  const joint = detail?.distance.joint_quality ?? 0;
  const barWidth = `${Math.max(0, Math.min(100, joint * 100)).toFixed(1)}%`;
  const evidence = detail?.inputs.point_evidence;
  const guidance = detail?.inputs.local_guidance;
  const featured = useMemo(() => cases.filter((item) => item.featured), [cases]);
  const extentLabel: Record<string, string> = { plate: "板", rod: "杆", block: "块" };
  const generatorLabel: Record<string, string> = {
    extrude_cut: "拉伸后切割",
    revolve_along_long: "沿长轴旋转/圆柱",
    sweep_or_revolve_in_long_plane: "在两长轴平面内扫掠或旋转",
    block_or_loft: "块或放样",
  };

  return (
    <div className="page compare-page">
      <div className="compare-top">
        <div>
          <h1>生成件对照真值</h1>
          <p className="subtitle">看输入长什么样，以及生成的 CAD 离原来的零件有多远。不展示执行流水账。</p>
        </div>
        <a className="ghost-link" href="/harness-v2.html?debug=1">
          打开执行调试
        </a>
      </div>

      <div className="panel compare-toolbar">
        <label>
          零件
          <select value={stem} onChange={(event) => setStem(event.target.value)}>
            {featured.length > 0 && (
              <optgroup label="典型对照">
                {featured.map((item) => (
                  <option key={`feat-${item.stem}`} value={item.stem}>
                    {item.family} · {item.label} · {fmt(item.joint_quality)}
                  </option>
                ))}
              </optgroup>
            )}
            <optgroup label="V5 其余零件">
              {cases.filter((item) => !item.featured).map((item) => (
                <option key={item.stem} value={item.stem}>
                  {item.family} · {fmt(item.joint_quality)} · {item.difficulty}
                </option>
              ))}
            </optgroup>
          </select>
        </label>
        <label>
          条件
          <select value={condition} onChange={(event) => setCondition(event.target.value)}>
            <option value="I1P_geom">照片 + 点云几何</option>
            <option value="I1">仅照片</option>
            <option value="P_geom">仅点云几何</option>
          </select>
        </label>
        <div className="mode-toggle">
          <button className={mode === "side" ? "active" : ""} onClick={() => setMode("side")}>
            并排
          </button>
          <button className={mode === "overlay" ? "active" : ""} onClick={() => setMode("overlay")}>
            叠在一起
          </button>
        </div>
      </div>

      {error && <p className="error">{error}</p>}
      {loading && <p className="hint">正在加载对照…</p>}

      {detail && (
        <>
          <div className="panel">
            <div className="compare-score-row">
              <div>
                <div className="compare-kicker">{detail.family}</div>
                <h2 className="compare-title">{detail.distance.label}</h2>
                <p className="hint">
                  {detail.condition_note}。本条件实际送给模型的是：
                  {detail.inputs.images_sent ? " 照片" : ""}
                  {detail.inputs.point_geom_sent ? " 点云几何事实" : ""}
                  {detail.inputs.text_sent ? " 文本" : ""}
                  {!detail.inputs.images_sent && !detail.inputs.point_geom_sent && !detail.inputs.text_sent
                    ? " （无模态记录）"
                    : ""}
                  。
                </p>
              </div>
              <div className="compare-stats">
                <div>
                  <span>像真值的程度</span>
                  <strong>{fmt(detail.distance.joint_quality)}</strong>
                </div>
                <div>
                  <span>表面距离 CD</span>
                  <strong>{fmt(detail.distance.shape_only_cd)}</strong>
                </div>
                <div>
                  <span>体积重叠 IoU</span>
                  <strong>{fmt(detail.distance.voxel_iou)}</strong>
                </div>
              </div>
            </div>
            <div className="compare-bar">
              <div style={{ width: barWidth }} />
            </div>
            <p className="hint">
              包围盒（归一化）真值 {sizeText(detail.distance.gt_size)}，生成 {sizeText(detail.distance.pred_size)}。
              蓝色是真值，橙色是模型。分数越高、表面距离越小，看起来越近。
            </p>
            <DualCadViewer gtUrl={detail.assets.gt_stl} predUrl={detail.assets.pred_stl} mode={mode} />
          </div>

          <div className="compare-input-grid">
            <div className="panel">
              <h2>输入照片{detail.inputs.images_sent ? "" : "（此条件未送给模型）"}</h2>
              <div className="compare-thumbs">
                {detail.assets.images.map((src, index) => (
                  <figure key={src}>
                    <img src={src} alt={IMAGE_LABELS[index]} />
                    <figcaption>{IMAGE_LABELS[index]}</figcaption>
                  </figure>
                ))}
              </div>
            </div>
            <div className="panel">
              <h2>点云投影{detail.inputs.point_geom_sent ? "（几何事实来自这朵云）" : ""}</h2>
              <div className="compare-thumbs">
                {detail.assets.pointcloud_images.map((src, index) => (
                  <figure key={src}>
                    <img src={src} alt={IMAGE_LABELS[index]} />
                    <figcaption>{IMAGE_LABELS[index]}</figcaption>
                  </figure>
                ))}
              </div>
              {points.length > 0 && (
                <>
                  <p className="hint">{points.length} 点，已放到和评分相同的归一化框里。</p>
                  <PointCloudViewer points={points} />
                </>
              )}
            </div>
          </div>

          <div className="panel">
            <h2>本地引导</h2>
            <p className="hint">与 v5 提示词同源：四条构造决策来自当前任务点云，不写零件族名。</p>
            {guidance ? (
              <ul className="compare-facts">
                <li>
                  工作面 {guidance.workplane}（薄轴 {guidance.thin_axis}，长轴 {guidance.long_axis}）
                </li>
                <li>
                  外形 {extentLabel[guidance.extent_class] ?? guidance.extent_class}
                  （最小/最长边 {fmt(guidance.thin_ratio, 3)}）
                </li>
                <li>
                  生成元 {generatorLabel[guidance.generator ?? ""] ?? guidance.generator ?? "—"}
                  {guidance.generator_verbs?.length ? `（${guidance.generator_verbs.join(" / ")}）` : ""}
                </li>
                <li>
                  剖面 {guidance.topology === "hollow" ? "空心" : guidance.topology === "unmeasured" ? "未测" : "实心"}
                  {guidance.inner_radius_known
                    ? `，内径 ${fmt(guidance.inner_radius)}${guidance.outer_radius != null ? ` / 外径 ${fmt(guidance.outer_radius)}` : ""}`
                    : guidance.topology === "unmeasured"
                      ? "（此条件没有点云截面）"
                      : "，内径未测到（不要编壁厚）"}
                </li>
                <li>包围盒 {guidance.bbox_size.map((item) => item.toFixed(3)).join(" × ")}</li>
                {guidance.loft_if_section_scales_differ ? <li>两截面外廓尺度不同，可 loft</li> : null}
              </ul>
            ) : (
              <p className="hint">没有点云证据，无法计算姿态。</p>
            )}
          </div>

          <div className="compare-input-grid">
            <div className="panel">
              <h2>点云告诉模型的事实</h2>
              {evidence?.bbox_size ? (
                <ul className="compare-facts">
                  <li>包围盒比例 {evidence.bbox_size.map((item) => item.toFixed(3)).join(" × ")}</li>
                  <li>{evidence.point_count ?? "—"} 个采样点</li>
                  {Object.entries(evidence.sections ?? {}).map(([axis, block]) => (
                    <li key={axis}>
                      {axis} 截面外轮廓 {block.bbox_size?.map((item) => item.toFixed(3)).join(" × ") ?? "—"}
                    </li>
                  ))}
                  {(evidence.symmetry ?? []).map((item, index) => (
                    <li key={`${item.type}-${index}`}>
                      对称 {item.type}，把握 {fmt(item.confidence)}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="hint">没有点云几何事实。</p>
              )}
            </div>
            <div className="panel">
              <h2>零件文本{detail.inputs.text_sent ? "" : "（此条件未送给模型，仅供你对照）"}</h2>
              <p className="compare-l1">{detail.inputs.l1 || "无 L1"}</p>
              <pre className="compare-l3">{detail.inputs.l3 || "无 L3"}</pre>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
