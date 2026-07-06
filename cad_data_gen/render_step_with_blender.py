#!/usr/bin/env python3
"""Blender-side STEP/STL renderer used by ``build_step_assets``.

The script is executed inside Blender:

    blender -b --python render_step_with_blender.py -- --config config.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(description="Render STEP views in Blender.")
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _objects_before() -> set[str]:
    return {obj.name for obj in bpy.context.scene.objects}


def _new_mesh_objects(before: set[str]) -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.name not in before and obj.type == "MESH"]


def import_step(step_path: str) -> list[bpy.types.Object]:
    before = _objects_before()
    errors: list[str] = []
    operators = [
        ("bpy.ops.wm.step_import", lambda: bpy.ops.wm.step_import(filepath=step_path)),
        ("bpy.ops.import_scene.step", lambda: bpy.ops.import_scene.step(filepath=step_path)),
        ("bpy.ops.import_mesh.step", lambda: bpy.ops.import_mesh.step(filepath=step_path)),
    ]
    for name, op in operators:
        try:
            op()
            objects = _new_mesh_objects(before)
            if objects:
                return objects
            errors.append(f"{name}: imported no mesh objects")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("No usable STEP import operator found. Tried: " + "; ".join(errors))


def convert_step_to_stl_with_freecad(step_path: str) -> str:
    out_file = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    out_path = out_file.name
    out_file.close()
    script_file = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    script_file.write(
        "\n".join(
            [
                "import sys",
                "import FreeCAD",
                "import Import",
                "import Mesh",
                "step_path = sys.argv[-2]",
                "out_path = sys.argv[-1]",
                "doc = FreeCAD.newDocument('step_to_stl')",
                "Import.insert(step_path, doc.Name)",
                "doc.recompute()",
                "objects = [obj for obj in doc.Objects if hasattr(obj, 'Shape')]",
                "if not objects:",
                "    raise RuntimeError('FreeCAD imported no shape objects')",
                "Mesh.export(objects, out_path)",
                "FreeCAD.closeDocument(doc.Name)",
            ]
        )
    )
    script_file.close()
    try:
        subprocess.run(
            ["freecadcmd", script_file.name, step_path, out_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        Path(script_file.name).unlink(missing_ok=True)
    if not Path(out_path).is_file() or Path(out_path).stat().st_size == 0:
        raise RuntimeError(f"FreeCAD did not produce a valid STL: {out_path}")
    return out_path


def import_stl(stl_path: str) -> list[bpy.types.Object]:
    before = _objects_before()
    errors: list[str] = []
    operators = [
        ("bpy.ops.wm.stl_import", lambda: bpy.ops.wm.stl_import(filepath=stl_path)),
        ("bpy.ops.import_mesh.stl", lambda: bpy.ops.import_mesh.stl(filepath=stl_path)),
    ]
    for name, op in operators:
        try:
            op()
            objects = _new_mesh_objects(before)
            if objects:
                return objects
            errors.append(f"{name}: imported no mesh objects")
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("Blender STL import failed. Tried: " + "; ".join(errors))


def import_step_or_fallback(config: dict[str, Any]) -> list[bpy.types.Object]:
    mesh_path = config.get("mesh_path")
    if mesh_path:
        return import_stl(str(mesh_path))
    try:
        return import_step(str(config["step_path"]))
    except Exception:
        stl_path = convert_step_to_stl_with_freecad(str(config["step_path"]))
        return import_stl(stl_path)


def _set_node_input(node: bpy.types.Node, names: tuple[str, ...], value: Any) -> None:
    for name in names:
        if name in node.inputs:
            node.inputs[name].default_value = value
            return


def make_material(name: str, color: list[float], roughness: float = 0.58) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = tuple(color)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        _set_node_input(bsdf, ("Base Color",), tuple(color))
        _set_node_input(bsdf, ("Roughness",), float(roughness))
    return material


def make_emission_material(name: str, color: list[float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = tuple(color)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission")
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    _set_node_input(emission, ("Color",), tuple(color))
    _set_node_input(emission, ("Strength",), 1.0)
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def normalize_objects(objects: list[bpy.types.Object], raw_center: list[float], scale: float) -> None:
    center = Vector((float(raw_center[0]), float(raw_center[1]), float(raw_center[2])))
    transform = (
        Matrix.Translation(Vector((0.0, 0.0, 0.0)))
        @ Matrix.Diagonal((float(scale), float(scale), float(scale), 1.0))
        @ Matrix.Translation(-center)
    )
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world


def view_basis(front: list[float]) -> tuple[Vector, Vector, Vector]:
    direction = Vector(front).normalized()
    up = Vector((0.0, 1.0, 0.0))
    right = up.cross(direction)
    if right.length < 1e-8:
        up = Vector((0.0, 0.0, 1.0))
        right = up.cross(direction)
    right.normalize()
    true_up = direction.cross(right)
    true_up.normalize()
    return right, true_up, direction


def projected_span(front: list[float]) -> float:
    right, true_up, _ = view_basis(front)
    corners = [
        Vector((x, y, z)) - Vector((0.0, 0.0, 0.0))
        for x in (-0.5, 0.5)
        for y in (-0.5, 0.5)
        for z in (-0.5, 0.5)
    ]
    xs = [corner.dot(right) for corner in corners]
    ys = [corner.dot(true_up) for corner in corners]
    return max(max(xs) - min(xs), max(ys) - min(ys))


def setup_camera(front: list[float], camera_distance: float, ortho_fill: float = 0.76) -> bpy.types.Object:
    lookat = Vector((0.0, 0.0, 0.0))
    direction = Vector(front).normalized()
    distance = max(abs(float(camera_distance)), 2.4)
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = lookat - direction * distance
    cam.rotation_euler = (lookat - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = projected_span(front) / float(ortho_fill)
    cam.data.clip_start = 0.01
    cam.data.clip_end = 100.0
    bpy.context.scene.camera = cam
    return cam


def add_foreground_occluder(camera: bpy.types.Object, front: list[float], config: dict[str, Any], view_index: int) -> bpy.types.Object | None:
    foreground = config.get("foreground_occluder")
    if not foreground:
        return None
    views = foreground.get("views") or []
    view_config = views[view_index] if view_index < len(views) else {}
    offset_x, offset_y = view_config.get("offset_xy", [0.0, 0.0])
    size_x, size_y = view_config.get("size_xy", [0.28, 0.28])
    depth = float(view_config.get("depth", foreground.get("depth", 0.45)))
    right, true_up, direction = view_basis(front)
    distance = (Vector((0.0, 0.0, 0.0)) - camera.location).length
    center = camera.location + direction * distance * min(max(depth, 0.05), 0.95)
    center += right * float(offset_x) * camera.data.ortho_scale
    center += true_up * float(offset_y) * camera.data.ortho_scale
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = bpy.context.object
    obj.name = f"ForegroundOccluder_{view_index:03d}"
    obj.dimensions = (
        max(1e-4, float(size_x) * camera.data.ortho_scale),
        max(1e-4, float(size_y) * camera.data.ortho_scale),
        0.01,
    )
    obj.rotation_euler = camera.rotation_euler
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    color = foreground.get("color", [1.0, 0.62, 0.08, 1.0])
    material = bpy.data.materials.get("ForegroundOccluderMaterial") or make_emission_material(
        "ForegroundOccluderMaterial",
        color,
    )
    obj.data.materials.append(material)
    return obj


def setup_lighting(config: dict[str, Any]) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    lighting = config.get("lighting", {}) or {}
    world.color = tuple([float(lighting.get("world", 0.28))] * 3)
    bpy.ops.object.light_add(type="AREA", location=(2.5, -3.0, 4.0))
    key = bpy.context.object
    key.name = "KeyArea"
    key.data.energy = float(lighting.get("key_energy", 150.0))
    key.data.size = 4.0
    bpy.ops.object.light_add(type="AREA", location=(-3.0, 2.0, 3.0))
    fill = bpy.context.object
    fill.name = "FillArea"
    fill.data.energy = float(lighting.get("fill_energy", 35.0))
    fill.data.size = 5.0


def setup_render(config: dict[str, Any]) -> None:
    scene = bpy.context.scene
    scene.render.resolution_x = int(config["img_size"])
    scene.render.resolution_y = int(config["img_size"])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    desired_engine = str(config.get("engine", "CYCLES"))
    for engine in (desired_engine, "CYCLES", "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(config.get("samples", 64))
    try:
        scene.view_settings.view_transform = "Standard"
    except TypeError:
        pass


def render_views(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, front in enumerate(config["fronts"]):
        old_camera = bpy.context.scene.camera
        if old_camera is not None:
            bpy.data.objects.remove(old_camera, do_unlink=True)
        camera = setup_camera(front, float(config.get("camera_distance", -0.9)))
        occluder = add_foreground_occluder(camera, front, config, i)
        try:
            bpy.context.scene.render.filepath = str(output_dir / f"view_{i:03d}.png")
            bpy.ops.render.render(write_still=True)
        finally:
            if occluder is not None:
                mesh = occluder.data
                bpy.data.objects.remove(occluder, do_unlink=True)
                if mesh.users == 0:
                    bpy.data.meshes.remove(mesh)


def main() -> None:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    clear_scene()
    objects = import_step_or_fallback(config)
    normalize_objects(objects, config["raw_center"], float(config["scale"]))
    material = make_material("ModelMaterial", config.get("model_color", [0.45, 0.68, 0.95, 1.0]))
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
    setup_lighting(config)
    setup_render(config)
    render_views(config)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[render_step_with_blender] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
