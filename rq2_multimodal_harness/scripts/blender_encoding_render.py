"""Headless Blender renderer for the frozen I1/I2 visual encodings.

Usage:
  blender -b -P blender_encoding_render.py -- --obj part.obj --outdir cache \
    --encoding I1 --size 512 --padding 0.06 --views front,side,top,isometric
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


CAMERAS = {
    "front": (0.0, -1.0, 0.0),
    "side": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "isometric": (1.0, -1.0, 1.0),
}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--encoding", required=True, choices=("I1", "I2"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--padding", type=float, default=0.06)
    parser.add_argument("--views", default="front,side,top,isometric")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for blocks in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for block in list(blocks):
            blocks.remove(block)


def import_obj(path: Path) -> list:
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(
            filepath=str(path),
            forward_axis="Y",
            up_axis="Z",
        )
    else:
        bpy.ops.import_scene.obj(
            filepath=str(path),
            axis_forward="Y",
            axis_up="Z",
        )
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def canonicalize_objects(objects: list) -> None:
    points = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    low = Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = Vector(tuple(max(point[i] for point in points) for i in range(3)))
    center = (low + high) / 2
    longest = max(high - low)
    if longest <= 1e-12:
        raise ValueError("OBJ bbox 尺度为零")
    transform = Matrix.Scale(1.0 / longest, 4) @ Matrix.Translation(-center)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()


def setup_world() -> None:
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 0.8


def add_area_light(name: str, location: tuple[float, float, float], energy: float, size: float) -> None:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()


def i1_material() -> object:
    material = bpy.data.materials.new("I1Studio")
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (0.55, 0.60, 0.68, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.42
    bsdf.inputs["Metallic"].default_value = 0.0
    return material


def i2_material() -> object:
    """Object-space normal encoded exactly as RGB=(n+1)/2."""
    material = bpy.data.materials.new("I2ObjectNormals")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    transform = nodes.new("ShaderNodeVectorTransform")
    transform.vector_type = "NORMAL"
    transform.convert_from = "WORLD"
    transform.convert_to = "OBJECT"
    scale = nodes.new("ShaderNodeVectorMath")
    scale.operation = "SCALE"
    scale.inputs[3].default_value = 0.5
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    links.new(geometry.outputs["Normal"], transform.inputs["Vector"])
    links.new(transform.outputs["Vector"], scale.inputs[0])
    links.new(scale.outputs["Vector"], add.inputs[0])
    links.new(add.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def setup_camera(padding: float) -> object:
    data = bpy.data.cameras.new("Camera")
    data.type = "ORTHO"
    # Canonical longest edge is 1; fixed frame includes configured padding per side.
    data.ortho_scale = 1.0 + 2.0 * padding
    data.clip_start = 0.01
    data.clip_end = 20.0
    camera = bpy.data.objects.new("Camera", data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    return camera


def main() -> None:
    args = parse_args()
    views = tuple(args.views.split(","))
    unknown = set(views) - set(CAMERAS)
    if unknown:
        raise ValueError(f"未知相机: {sorted(unknown)}")
    obj_path = Path(args.obj).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    setup_world()
    objects = import_obj(obj_path)
    if not objects:
        raise RuntimeError(f"OBJ 中没有 mesh: {obj_path}")
    canonicalize_objects(objects)
    material = i1_material() if args.encoding == "I1" else i2_material()
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.use_freestyle = False  # I1 explicitly has no Freestyle overlay.
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.view_settings.look = "None"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    if args.encoding == "I1":
        add_area_light("Key", (3.5, -4.0, 5.0), 800.0, 4.0)
        add_area_light("Fill", (-4.0, -2.0, 2.5), 350.0, 5.0)
        add_area_light("Rim", (1.0, 4.0, 4.0), 500.0, 3.0)

    camera = setup_camera(args.padding)
    for view in views:
        direction = Vector(CAMERAS[view]).normalized()
        camera.location = direction * 4.0
        camera.rotation_euler = (-camera.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        scene.render.filepath = str(outdir / f"{obj_path.stem}__{args.encoding}__{view}.png")
        bpy.ops.render.render(write_still=True)
    print(f"ENCODING_RENDER_DONE encoding={args.encoding} views={len(views)}")


if __name__ == "__main__":
    main()
