from __future__ import annotations

import numpy as np
import trimesh


def mesh_to_point_cloud_with_normals(mesh: trimesh.Trimesh, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample fixed-size surface points and their face normals from a mesh."""
    points, face_indices = trimesh.sample.sample_surface(mesh, int(num_points))
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    normals = face_normals[np.asarray(face_indices, dtype=np.int64)]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(lengths, 1e-12)
    return np.asarray(points, dtype=np.float32), np.asarray(normals, dtype=np.float32)


def mesh_to_point_cloud(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Sample a fixed-size surface point cloud from a mesh."""
    points, _ = mesh_to_point_cloud_with_normals(mesh, num_points)
    return points
