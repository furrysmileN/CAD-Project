"""阶段 2：PointEvidence 离线质量审计（零 API）。

仅评分阶段使用 GT STEP 检查证据准确率：bbox 尺寸、主轴对齐、主平面、
截面外环、对称候选。GT 不进入证据生成路径（service/evidence 不接受 GT 路径）。

门禁：bbox/主轴/截面在多数样本上可靠才允许进入 API 实验。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import read_jsonl
from .pointcloud.compare import sample_step
from .pointcloud.evidence import build_evidence
from .pointcloud.io import clean_points, load_point_cloud
from .pointcloud.sections import adaptive_thickness, query_cross_section

SECTION_AXES = ("XY", "XZ", "YZ")
_SECTION_PLANES = {
    "XY": np.array([0.0, 0.0, 1.0]),
    "XZ": np.array([0.0, 1.0, 0.0]),
    "YZ": np.array([1.0, 0.0, 0.0]),
}
BOUNDS = {
    "bbox_relative_error": 0.10,
    "axis_alignment": 0.95,
    "section_relative_error": 0.20,
    "symmetry": 0.5,
}


def _gt_geometry(step_path: Path, n_points: int = 8192, seed: int = 42) -> dict[str, Any]:
    points = sample_step(step_path, n_points=n_points, seed=seed)
    low = points.min(axis=0)
    high = points.max(axis=0)
    size = high - low
    center = (low + high) / 2.0
    scale = float(size.max())
    canonical = (points - center) / scale
    cov = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order].T
    return {
        "points": points,
        "canonical": canonical,
        "bbox_size": (high - low).tolist(),
        "canonical_bbox_size": (canonical.max(axis=0) - canonical.min(axis=0)).tolist(),
        "axes": axes,
        "eigenvalue_ratios": (eigenvalues[order] / max(float(eigenvalues[order][0]), 1e-12)).tolist(),
        "center": center,
        "scale": scale,
        "step_path": step_path,
    }


def _gt_mesh(gt: dict[str, Any]):
    """GT STEP 的 canonical 帧 trimesh 网格（评分阶段）。

    与证据同一坐标约定：顶点按 GT bbox 中心平移、最长边归一，
    使 GT 截面平面（过 canonical 原点）与证据截面平面完全对齐。
    """
    from .geometry import _shape_mesh
    import cadquery as cq

    shape = cq.importers.importStep(str(gt["step_path"])).val()
    return _shape_mesh(shape, gt["center"], gt["scale"])


def _gt_section_exact(mesh, normal: np.ndarray) -> list[float] | None:
    """trimesh 平面截面外环 bbox（canonical 帧）。"""
    try:
        import trimesh

        path = mesh.section(plane_origin=[0.0, 0.0, 0.0], plane_normal=normal.tolist())
    except Exception:
        return None
    if path is None:
        return None
    bounds = np.asarray(path.bounds)
    if bounds.ndim != 2 or bounds.shape != (2, 3):
        return None
    section_2d = np.abs(bounds[1] - bounds[0])
    # 截面平面法向上的跨度应为 0；返回平面内两个跨度
    normal_abs = np.abs(normal)
    in_plane = section_2d[normal_abs < 0.99]
    if len(in_plane) != 2:
        return None
    return in_plane.tolist()


# 主轴对齐：只在「方向唯一」的 PCA 轴上打分。
# 两条特征值比接近的轴构成近退化子空间（圆盘面内、回转件径向平面），方向任意，双方都排除。
DEGENERATE_RATIO = 0.85


def _well_defined_pca_axes(
    axes: list,
    ratios: list[float] | None,
    *,
    ratio_threshold: float = 0.1,
    pair_ratio: float = DEGENERATE_RATIO,
) -> list[np.ndarray]:
    """留下方向唯一的 PCA 轴。

    特征值已按最大特征值归一。若两条轴的 min/max ≥ pair_ratio，视为同一退化子空间，
    两条都不打分。比值 < ratio_threshold 的近一维轴也不打分。
    回转体因此只保留回转轴；圆盘则可能全部不适用（返回空）。
    """
    if not axes:
        return []
    ratio_list = list(ratios or [])
    if len(ratio_list) < len(axes):
        ratio_list.extend([0.0] * (len(axes) - len(ratio_list)))
    keep = [float(ratio_list[index]) >= ratio_threshold for index in range(len(axes))]
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            a = float(ratio_list[i])
            b = float(ratio_list[j])
            if min(a, b) < ratio_threshold:
                continue
            if min(a, b) / max(a, b) >= pair_ratio:
                keep[i] = False
                keep[j] = False
    defined: list[np.ndarray] = []
    for axis, flag in zip(axes, keep):
        if not flag:
            continue
        vector = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            continue
        defined.append(vector / norm)
    return defined


def _axis_alignment(
    evidence_axes: list[list[float]],
    evidence_ratios: list[float] | None,
    gt: dict[str, Any],
    ratio_threshold: float = 0.1,
    degenerate_ratio: float = DEGENERATE_RATIO,
) -> float | None:
    """证据主轴与 GT 主轴的对齐（贪心匹配）。

    只在双方都「方向唯一」的轴上打分：成对的近等特征值子空间（径向平面、圆盘面）排除。
    全退化时返回 None。
    """
    gt_axes = _well_defined_pca_axes(
        list(gt["axes"]),
        list(gt.get("eigenvalue_ratios") or []),
        ratio_threshold=ratio_threshold,
        pair_ratio=degenerate_ratio,
    )
    evidence_axes_defined = _well_defined_pca_axes(
        evidence_axes,
        evidence_ratios,
        ratio_threshold=ratio_threshold,
        pair_ratio=degenerate_ratio,
    )
    if not gt_axes or not evidence_axes_defined:
        return None
    used = [False] * len(gt_axes)
    total = 0.0
    matched = 0
    for axis in evidence_axes_defined:
        scores = [
            abs(float(np.dot(axis, target))) if not used[index] else -1.0
            for index, target in enumerate(gt_axes)
        ]
        best = int(np.argmax(scores))
        if scores[best] >= 0:
            total += max(scores[best], 0.0)
            used[best] = True
            matched += 1
    if not matched:
        return None
    return float(total / matched)


def _section_outer(
    evidence: dict[str, Any],
    gt: dict[str, Any],
    mesh,
    gt_canonical: np.ndarray,
    thickness: float,
) -> dict[str, Any]:
    """截面外环 bbox：证据（2048 点带内）与密集 GT 点云同一厚度带内直接对比。

    GT 参考使用点云带内切片（同厚度）而不是 trimesh 精确截面：点云截面语义
    是「平面附近厚度带内的表面点」，与精确几何截面在薄/倾斜结构上可能不同。
    精确截面仍作为诊断参考（exact_size）输出。
    """
    result: dict[str, Any] = {}
    for axis in SECTION_AXES:
        evidence_outer = (evidence.get("sections") or {}).get(axis, {}).get("outer")
        gt_band = query_cross_section(
            gt_canonical, np.zeros(3), _SECTION_PLANES[axis], thickness
        ).get("outer")
        gt_exact = _gt_section_exact(mesh, _SECTION_PLANES[axis])
        row: dict[str, Any] = {
            "evidence_size": evidence_outer["bbox_size"] if evidence_outer else None,
            "gt_band_size": gt_band["bbox_size"] if gt_band else None,
            "gt_exact_size": gt_exact,
            "thickness": thickness,
        }
        if evidence_outer and gt_band:
            relative = [
                abs(ev - ref) / max(ref, 1e-12)
                for ev, ref in zip(evidence_outer["bbox_size"], gt_band["bbox_size"])
            ]
            row["relative_error"] = float(np.mean(relative))
            row["max_relative_error"] = float(np.max(relative))
        else:
            row["relative_error"] = None
            row["max_relative_error"] = None
        result[axis] = row
    return result


def _symmetry_check(evidence: dict[str, Any], gt: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    """把证据的对称候选直接拿到 GT 点云上验证反射支持。

    - GT 存在强对称（PCA 轴平面支持 ≥ 0.5）：证据 top 候选在 GT 上的支持
      也 ≥ 0.5 → 命中（真实对称面，方向可不同，因为任意过轴的平面都有效）。
    - GT 无强对称：证据 top 支持 < 0.5 → 无过度声称。
    """
    rng = np.random.default_rng(seed)
    sampled = gt["canonical"][rng.choice(len(gt["canonical"]), size=512, replace=False)]
    from scipy.spatial import cKDTree

    tree = cKDTree(gt["canonical"])
    gt_candidates = []
    for axis in gt["axes"]:
        reflected = sampled - 2.0 * (sampled @ axis)[:, None] * axis
        distances, _ = tree.query(reflected, k=1)
        gt_candidates.append({"normal": axis.tolist(), "support": float((distances <= 0.015).mean())})
    gt_best = max(gt_candidates, key=lambda item: item["support"])
    has_gt_symmetry = gt_best["support"] >= 0.5
    evidence_top = (evidence.get("symmetry_candidates") or [None])[0]
    if evidence_top is None:
        return {
            "has_gt_symmetry": has_gt_symmetry,
            "gt_best_support": gt_best["support"],
            "evidence_support": None,
            "hit": not has_gt_symmetry,
            "note": "no_evidence_candidate",
        }
    evidence_normal = np.asarray(evidence_top["normal"])
    reflected = sampled - 2.0 * (sampled @ evidence_normal)[:, None] * evidence_normal
    distances, _ = tree.query(reflected, k=1)
    ev_gt_support = float((distances <= 0.015).mean())
    if has_gt_symmetry:
        hit = ev_gt_support >= 0.5
        note = "ok" if hit else "weak_or_wrong_plane"
    else:
        hit = (evidence_top.get("support_ratio") or 0.0) < 0.5
        note = "ok" if hit else "overclaim"
    return {
        "has_gt_symmetry": has_gt_symmetry,
        "gt_best_support": gt_best["support"],
        "gt_best_normal": gt_best["normal"],
        "evidence_support": evidence_top.get("support_ratio"),
        "evidence_normal": evidence_top["normal"],
        "ev_gt_support": ev_gt_support,
        "hit": hit,
        "note": note,
    }


def audit_evidence(
    sample_ids: list[str],
    manifest_path: Path,
    pointcloud_root: Path,
    evidence_dir: Path,
    *,
    density: int = 2048,
    gt_n_points: int = 8192,
    gt_seed: int = 42,
    section_thickness: float | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    manifest = {row["sample_id"]: row for row in read_jsonl(manifest_path)}
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        row = manifest.get(sample_id)
        if row is None:
            raise RuntimeError(f"manifest 中缺少样本 {sample_id}")
        npy_path = pointcloud_root / str(density) / f"{sample_id}.npy"
        if not npy_path.is_file():
            raise RuntimeError(f"缺少点云 {npy_path}")
        try:
            raw = load_point_cloud(npy_path)
            cloud, _quality = clean_points(raw)
            cloud_canonical = (cloud - cloud.mean(axis=0)) / float(
                np.max(cloud.max(axis=0) - cloud.min(axis=0))
            )
            # 自适应厚度：按输入点云密度决定，同一厚度同时用于证据带与 GT 密集带，
            # 保证对比语义一致（section_thickness=None 时）。
            effective_thickness = (
                section_thickness
                if section_thickness is not None
                else adaptive_thickness(cloud_canonical)
            )
            evidence = build_evidence(
                npy_path,
                section_thickness=effective_thickness,
                config={"audit": "offline"},
            )
            gt_step = Path(row["step"]["path"])
            gt = _gt_geometry(gt_step, n_points=gt_n_points, seed=gt_seed)
            mesh = _gt_mesh(gt)
            gt_canonical = gt["canonical"]

            bbox_relative = [
                abs(ev - ref) / max(ref, 1e-12)
                for ev, ref in zip(
                    evidence["frame"]["bbox_size"],
                    (gt_canonical.max(axis=0) - gt_canonical.min(axis=0)).tolist(),
                )
            ]
            axis_alignment = _axis_alignment(
                evidence["frame"]["principal_axes"],
                evidence["frame"].get("eigenvalue_ratios"),
                gt,
            )
            sections = _section_outer(
                evidence, gt, mesh, gt_canonical, thickness=effective_thickness
            )
            symmetry = _symmetry_check(evidence, gt, seed=gt_seed)

            plane = (evidence.get("primitive_candidates") or [None])[0]
            plane_alignment = (
                max(abs(float(np.dot(np.asarray(plane["normal"]), axis))) for axis in gt["axes"])
                if plane
                else None
            )

            rows.append(
                {
                    "sample_id": sample_id,
                    "family": row.get("family"),
                    "difficulty": row.get("difficulty"),
                    "bbox_relative_error": float(np.mean(bbox_relative)),
                    "bbox_max_relative_error": float(np.max(bbox_relative)),
                    "bbox_axis_errors": [float(value) for value in bbox_relative],
                    "bbox_pass": float(np.max(bbox_relative)) <= BOUNDS["bbox_relative_error"],
                    "axis_alignment": axis_alignment,
                    "axis_pass": axis_alignment is None or axis_alignment >= BOUNDS["axis_alignment"],
                    "plane_alignment": plane_alignment,
                    "sections": sections,
                    "section_pass": all(
                        (item.get("max_relative_error") or 1.0) <= BOUNDS["section_relative_error"]
                        for item in sections.values()
                        if item.get("max_relative_error") is not None
                    ) and all(item.get("max_relative_error") is not None for item in sections.values()),
                    "symmetry": symmetry,
                    "symmetry_pass": bool(symmetry["hit"]),
                    "point_count": evidence["quality"]["point_count"],
                    "valid_ratio": evidence["quality"]["valid_ratio"],
                    "audit_error": None,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "sample_id": sample_id,
                    "family": row.get("family"),
                    "difficulty": row.get("difficulty"),
                    "bbox_relative_error": None,
                    "bbox_max_relative_error": None,
                    "bbox_axis_errors": [],
                    "bbox_pass": False,
                    "axis_alignment": None,
                    "axis_pass": False,
                    "plane_alignment": None,
                    "sections": {},
                    "section_pass": False,
                    "symmetry": {"hit": False},
                    "symmetry_pass": False,
                    "point_count": None,
                    "valid_ratio": None,
                    "audit_error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )

    n = len(rows)
    alignments = [row["axis_alignment"] for row in rows if row["axis_alignment"] is not None]
    bbox_errors = [row["bbox_relative_error"] for row in rows if row.get("bbox_relative_error") is not None]
    summary = {
        "n": n,
        "n_audit_error": sum(1 for row in rows if row.get("audit_error")),
        "bbox_pass_rate": sum(row["bbox_pass"] for row in rows) / n if n else None,
        "axis_pass_rate": sum(row["axis_pass"] for row in rows) / n if n else None,
        "section_pass_rate": sum(row["section_pass"] for row in rows) / n if n else None,
        "symmetry_pass_rate": sum(row["symmetry_pass"] for row in rows) / n if n else None,
        "mean_bbox_relative_error": float(np.mean(bbox_errors)) if bbox_errors else None,
        "mean_axis_alignment": float(np.mean(alignments)) if alignments else None,
        "axis_degenerate_samples": sum(1 for row in rows if row["axis_alignment"] is None),
    }
    gate = {
        "bbox_reliable": summary["bbox_pass_rate"] >= 0.9,
        "axes_reliable": summary["axis_pass_rate"] >= 0.9,
        "section_reliable": summary["section_pass_rate"] >= 0.75,
        "symmetry_reliable": summary["symmetry_pass_rate"] >= 0.75,
    }
    gate["passed"] = all(gate.values())
    report = {
        "schema_version": "rq2.pc.evidence_audit.v1",
        "density": density,
        "gt": {"n_points": gt_n_points, "seed": gt_seed, "note": "scoring-stage-only"},
        "thresholds": BOUNDS,
        "summary": summary,
        "gate": gate,
        "rows": rows,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evidence_audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        bbox_mean = (
            "NA"
            if summary["mean_bbox_relative_error"] is None
            else f"{summary['mean_bbox_relative_error']:.4f}"
        )
        axis_mean = (
            "NA" if summary["mean_axis_alignment"] is None else f"{summary['mean_axis_alignment']:.4f}"
        )
        lines = [
            f"# PointEvidence 离线质量审计（{n} 样本）",
            "",
            "> 零 API；GT STEP 仅评分阶段用于核对证据，不参与证据生成。",
            f"> 门禁阈值：bbox 每轴相对误差 ≤ {BOUNDS['bbox_relative_error']}、主轴对齐 ≥ {BOUNDS['axis_alignment']}、"
            f"截面外环相对误差 ≤ {BOUNDS['section_relative_error']}、对称命中（含真空通过）。",
            "",
            "## 汇总",
            "",
            f"- bbox 通过率：{summary['bbox_pass_rate']:.2f}（均值相对误差 {bbox_mean}）",
            f"- 主轴对齐通过率：{summary['axis_pass_rate']:.2f}（均值对齐 {axis_mean}）",
            f"- 截面通过率：{summary['section_pass_rate']:.2f}",
            f"- 对称通过率：{summary['symmetry_pass_rate']:.2f}",
            f"- 审计异常：{summary.get('n_audit_error') or 0}",
            "",
            f"**门禁：{'通过' if gate['passed'] else '未通过'}**（" + "，".join(
                f"{name}={'✓' if gate[name] else '✗'}" for name in ("bbox_reliable", "axes_reliable", "section_reliable", "symmetry_reliable")
            ) + "）",
            "",
            "## 逐样本",
            "",
            "| 样本 | family | bbox 最大相对误差 | 主轴对齐 | 截面通过 | 对称 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            if row.get("audit_error"):
                lines.append(
                    f"| {row['sample_id']} | {row['family']} | ERROR | ERROR | ✗ | {row['audit_error'][:80]} |"
                )
                continue
            sym = row["symmetry"]
            sym_label = f"{'✓' if row['symmetry_pass'] else '✗'} (GT支持 {sym.get('gt_best_support', 0):.2f}, 证据支持 {sym['evidence_support'] if sym.get('evidence_support') is not None else 'NA'})"
            axis_label = (
                f"{row['axis_alignment']:.3f}"
                if row["axis_alignment"] is not None
                else "NA(退化)"
            )
            lines.append(
                f"| {row['sample_id']} | {row['family']} | {row['bbox_max_relative_error']:.4f} | "
                f"{axis_label} | {'✓' if row['section_pass'] else '✗'} | {sym_label} |"
            )
        lines.append("")
        lines.append("截面逐轴明细与对称明细见 `evidence_audit.json`。")
        (output_dir / "EVIDENCE_AUDIT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
