from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, sha256_file, sha256_json
from .canonical import CanonicalTransform
from .evidence import build_evidence
from .io import clean_points, hash_points, load_point_cloud
from .tools import PointCloudSession


@dataclass
class PointCloudService:
    """实验期服务：prepare 阶段生成并缓存 PointEvidence + 运行期构造工具会话。

    泄漏约束：只读取任务输入点云；GT STEP / GT code 路径不允许传入。
    """

    def __init__(
        self,
        *,
        normals_k: int = 16,
        plane_tolerance: float = 0.012,
        plane_max_planes: int = 2,
        ransac_seed: int = 42,
        section_axes: tuple[str, ...] = ("XY", "XZ", "YZ"),
        section_thickness: float | None = None,
        symmetry_tolerance: float = 0.015,
        symmetry_sample: int = 512,
        n_points: int = 2048,
        seed: int = 42,
        version: str = "pointcloud.service.v1",
    ) -> None:
        self.params = {
            "normals_k": normals_k,
            "plane_tolerance": plane_tolerance,
            "plane_max_planes": plane_max_planes,
            "ransac_seed": ransac_seed,
            "section_axes": list(section_axes),
            "section_thickness": section_thickness,
            "symmetry_tolerance": symmetry_tolerance,
            "symmetry_sample": symmetry_sample,
            "n_points": n_points,
            "seed": seed,
            "version": version,
        }
        self.n_points = n_points
        self.seed = seed

    def config(self) -> dict[str, Any]:
        return dict(self.params)

    def evidence_path(self, evidence_dir: Path, sample_id: str) -> Path:
        return evidence_dir / f"{sample_id}.point_evidence.json"

    def prepare_evidence(
        self,
        npy_path: Path,
        evidence_dir: Path,
        sample_id: str,
        *,
        force: bool = False,
    ) -> dict[str, Any]:
        """生成（或复用缓存）PointEvidence。返回证据 dict。"""
        path = self.evidence_path(evidence_dir, sample_id)
        if path.is_file() and not force:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("config") == self.params and cached.get("source_sha256"):
                return cached
        evidence = build_evidence(npy_path, config=self.params)
        evidence["source_sha256"] = sha256_file(npy_path)
        evidence["config"] = self.params
        evidence["content_hash"] = sha256_json(evidence)
        atomic_write_json(path, evidence)
        return json.loads(path.read_text(encoding="utf-8"))

    def session(
        self,
        npy_path: Path,
        evidence: dict[str, Any] | None = None,
    ) -> PointCloudSession:
        """构造一次任务内的工具会话（运行期使用）。"""
        raw = load_point_cloud(npy_path)
        points, _quality = clean_points(raw)
        transform = CanonicalTransform.from_points(points)
        if evidence and evidence.get("cloud_id"):
            cloud_id = str(evidence["cloud_id"])
        else:
            cloud_id = "c_" + hash_points(points)[:16]
        return PointCloudSession(
            cloud_id=cloud_id,
            points_canonical=transform.forward(points),
            transform=transform,
            n_points=self.n_points,
            seed=self.seed,
        )
