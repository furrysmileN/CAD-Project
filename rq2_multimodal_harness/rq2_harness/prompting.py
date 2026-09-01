from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .common import sha256_json
from .conditions import modalities_for


SYSTEM_PROMPTS = {
    "v1": """You convert multimodal observations of one CAD part into a constrained HarnessCAD Plan.
Return exactly one JSON object and no commentary. Geometry uses a canonical frame: bbox center [0,0,0]
and longest bbox edge 1.0. The plan schema is harnesscad.plan.v1. Allowed primitives are box,
cylinder, sphere; allowed combine values are new, add, cut, intersect. The first operation is new.
Use 1-64 operations. Do not infer names, labels, provenance, classes, or metadata from the input.""",
    "v2": """You convert multimodal observations of one CAD part into a constrained HarnessCAD Plan.
Return exactly one JSON object and no commentary. Geometry uses a canonical frame: bbox center [0,0,0]
and longest bbox edge 1.0. The plan schema is harnesscad.plan.v2. Allowed op values are box, cylinder,
sphere, polygon_extrude, revolve_profile, hole, slot, transform, fillet, chamfer, linear_pattern.
Shape operations use combine=new/add/cut/intersect; only the first operation uses new. hole and slot
must use cut. polygon/profile points must explicitly close by repeating their first point. transform
and linear_pattern source must name an earlier shape operation. Use only global XY/XZ/YZ workplanes,
finite normalized coordinates, unit 3D axes, and 1-64 operations.

Every operation MUST contain exactly the required keys below:
box: id,op,combine,center,size
cylinder: id,op,combine,center,radius,height,axis
sphere: id,op,combine,center,radius
polygon_extrude: id,op,combine,workplane,points,depth,centered,offset
revolve_profile: id,op,combine,workplane,profile,axis,angle,offset
hole: id,op,combine,workplane,center,diameter,depth
slot: id,op,combine,workplane,center,length,width,depth,angle
transform: id,op,combine,source and at least one of translate or rotate
fillet: id,op,radius; optional edge_axis X/Y/Z
chamfer: id,op,distance; optional edge_axis X/Y/Z
linear_pattern: id,op,combine,source,direction,count,spacing

For hole/slot, center is always a 3-number global coordinate. For cylinder, axis is always a
3-number unit vector. Never omit a required key. If observations are ambiguous, prefer a simpler
valid approximation over an incomplete operation. Do not emit Python, metadata, comments, or extra keys.""",
    "v3": """You convert multimodal observations of one CAD part into a constrained HarnessCAD Plan.
Return exactly one JSON object and no commentary. Geometry uses a canonical frame: bbox center [0,0,0]
and longest bbox edge 1.0. The plan schema is harnesscad.plan.v2. Allowed op values are box, cylinder,
sphere, polygon_extrude, revolve_profile, hole, slot, transform, fillet, chamfer, linear_pattern.
Shape operations use combine=new/add/cut/intersect; only the first operation uses new. hole and slot
must use cut. polygon/profile points must explicitly close by repeating their first point. transform
and linear_pattern source must name an earlier shape operation. Use only global XY/XZ/YZ workplanes,
finite normalized coordinates, unit 3D axes, and 1-64 operations.

Every operation MUST contain exactly the required keys below:
box: id,op,combine,center,size
cylinder: id,op,combine,center,radius,height,axis
sphere: id,op,combine,center,radius
polygon_extrude: id,op,combine,workplane,points,depth,centered,offset
revolve_profile: id,op,combine,workplane,profile,axis,angle,offset
hole: id,op,combine,workplane,center,diameter,depth
slot: id,op,combine,workplane,center,length,width,depth,angle
transform: id,op,combine,source and at least one of translate or rotate
fillet: id,op,radius; optional edge_axis X/Y/Z
chamfer: id,op,distance; optional edge_axis X/Y/Z
linear_pattern: id,op,combine,source,direction,count,spacing

For hole/slot, center is always a 3-number global coordinate. For cylinder, axis is always a
3-number unit vector. Never omit a required key. If observations are ambiguous, prefer a simpler
valid approximation over an incomplete operation. Do not emit Python, metadata, comments, or extra keys.

Hard rules that reject plans (violating any of these fails validation):
- revolve_profile.axis MUST be exactly two DIFFERENT 2D points on the workplane, e.g. [[0,0],[1,0]].
  Two identical points such as [[0,0],[0,0]] are always invalid.
- revolve_profile.profile must lie entirely on ONE side of the axis; it must close by repeating its
  first point; use at least 4 points (first point repeated last).
- transform.rotate must be {origin: 3-number point [x,y,z], axis: unit 3-vector, angle: finite degrees}.
  origin is the 3-number global point the rotation axis passes through, NOT a 2D point.
- cylinder.axis and linear_pattern.direction must be unit 3-vectors with norm exactly 1, e.g. [0,0,1].
- polygon/profile vertices: no two consecutive vertices may be equal (except the closing repeat),
  edges must not cross (no self-intersection), and at least 3 distinct vertices are required.
- All numbers must be finite. linear_pattern.count must be an integer from 2 to 32. spacing must be
  a finite number in (0.0, 8.0].
- transform/linear_pattern source must reference the id of an EARLIER shape operation in this plan.""",
    "v4": """You convert multimodal observations of one CAD part into a constrained HarnessCAD Plan.
Return exactly one JSON object and no commentary. Geometry uses a canonical frame: bbox center [0,0,0]
and longest bbox edge 1.0. The plan schema is harnesscad.plan.v3. Allowed op values are box, cylinder,
sphere, polygon_extrude, revolve_profile, hole, slot, transform, fillet, chamfer, linear_pattern,
sweep_profile, loft_profiles. Use 1-128 operations. Prefer sweep_profile or loft_profiles for pipes,
springs, tapers, and gears instead of stacking boxes.

polygon_extrude / revolve_profile / sweep_profile: give exactly one of a closed points/profile list
(repeat the first point) OR a wire list. A wire starts with {kind:move,to:[u,v]} and continues with
{kind:line,to:[u,v]} or {kind:three_point_arc,through:[u,v],to:[u,v]}, and must close.

sweep_profile: exactly one of path (list of 3D points) or helix
{radius,pitch,turns,axis:unit3,center:3}. loft_profiles needs 2-8 profiles each with points or wire
plus offset. hole/slot must use cut. First combine is new.

Do not emit Python. If ambiguous, emit a valid simpler solid rather than an incomplete op.""",
    "v5": """You convert multimodal observations of one CAD part into a constrained HarnessCAD Plan.
Return exactly one JSON object and no commentary. Geometry uses a canonical frame: bbox center [0,0,0]
and longest bbox edge 1.0. The plan schema is harnesscad.plan.v3. Allowed op values are box, cylinder,
sphere, polygon_extrude, revolve_profile, hole, slot, transform, fillet, chamfer, linear_pattern,
sweep_profile, loft_profiles. Shape operations use combine=new/add/cut/intersect; only the first
operation uses new. hole and slot must use cut. polygon/profile points must explicitly close by
repeating their first point. transform and linear_pattern source must name an earlier shape operation.
Use only global XY/XZ/YZ workplanes, finite normalized coordinates, unit 3D axes, and 1-128 operations.

Every operation MUST contain exactly the required keys below:
box: id,op,combine,center,size
cylinder: id,op,combine,center,radius,height,axis
sphere: id,op,combine,center,radius
polygon_extrude: id,op,combine,workplane,points,depth,centered,offset
revolve_profile: id,op,combine,workplane,profile,axis,angle,offset
hole: id,op,combine,workplane,center,diameter,depth
slot: id,op,combine,workplane,center,length,width,depth,angle
transform: id,op,combine,source and at least one of translate or rotate
fillet: id,op,radius; optional edge_axis X/Y/Z
chamfer: id,op,distance; optional edge_axis X/Y/Z
linear_pattern: id,op,combine,source,direction,count,spacing
sweep_profile: id,op,combine,workplane,offset and exactly one of profile or wire, plus exactly one of path or helix
loft_profiles: id,op,combine,workplane,profiles (2-8 items, each with points or wire plus offset)

polygon_extrude / revolve_profile / sweep_profile: give exactly one of a closed points/profile list
(repeat the first point) OR a wire list. A wire starts with {kind:move,to:[u,v]} and continues with
{kind:line,to:[u,v]} or {kind:three_point_arc,through:[u,v],to:[u,v]}, and must close.
sweep_profile helix is {radius,pitch,turns,axis:unit3,center:3}.

For hole/slot, center is always a 3-number global coordinate. For cylinder, axis is always a
3-number unit vector. Never omit a required key. If observations are ambiguous, prefer a simpler
valid approximation over an incomplete operation. Do not emit Python, metadata, comments, or extra keys.

Hard rules that reject plans (violating any of these fails validation):
- revolve_profile.axis MUST be exactly two DIFFERENT 2D points on the workplane, e.g. [[0,0],[1,0]].
  Two identical points such as [[0,0],[0,0]] are always invalid.
- revolve_profile.profile must lie entirely on ONE side of the axis; it must close by repeating its
  first point; use at least 4 points (first point repeated last).
- transform.rotate must be {origin: 3-number point [x,y,z], axis: unit 3-vector, angle: finite degrees}.
  origin is the 3-number global point the rotation axis passes through, NOT a 2D point.
- cylinder.axis and linear_pattern.direction must be unit 3-vectors with norm exactly 1, e.g. [0,0,1].
- polygon/profile vertices: no two consecutive vertices may be equal (except the closing repeat),
  edges must not cross (no self-intersection), and at least 3 distinct vertices are required.
- All numbers must be finite. linear_pattern.count must be an integer from 2 to 32. spacing must be
  a finite number in (0.0, 8.0].
- transform/linear_pattern source must reference the id of an EARLIER shape operation in this plan.

Choose the generator from bbox proportions, not from part names. Prefer loft_profiles when
two sections have different outer scales. Do not revolve a large solid rectangle to fake a hollow.
Use only measured sizes; if an inner radius is absent, do not invent wall thickness.""",
    "v6": """You convert multimodal observations of one CAD part into a constrained HarnessCAD semantic plan.
Return exactly one JSON object and no commentary. Geometry uses a canonical frame: bbox center [0,0,0]
and longest bbox edge 1.0. The schema is harnesscad.plan.v3.1. Allowed operations are all Plan v3
operations plus sweep_path_ref.

When [DECISIONS] contains path_graph_status=resolved and the observed target follows that measured
path, emit sweep_path_ref with exactly: id,op,combine,evidence_ref. The evidence_ref must be one of
the listed refs. A deterministic binder will insert the measured profile and line/arc path; never copy
or alter its numeric values. This is preferred over inventing cylinders, a polyline, or a revolve.
If the image shows additional independent features, add ordinary Plan v3.1 operations after it.
If path evidence is unresolved or does not match the visible target, do not use sweep_path_ref.

For an explicit sweep_profile, provide workplane, offset, exactly one of profile/wire, exactly one of
path/path_wire/helix, and optional sweep_mode=fixed|frenet. A 3D path_wire starts with
{kind:move,to:[x,y,z]}, then line segments or
{kind:three_point_arc,through:[x,y,z],to:[x,y,z]}. Use only finite normalized coordinates.
Shape operations use combine=new/add/cut/intersect; only the first operation uses new. hole/slot use
cut. Use 1-128 operations. Never emit Python, comments, prose, family labels, or unlisted fields.""",
}

RAW_CADQUERY_SYSTEM = """You reconstruct one CAD part as executable CadQuery 2 Python.
Return only a Python script. Import cadquery as cq. Build a solid and bind it to the variable result.
Do not write files, do not import os/sys/subprocess, do not call show_object.
Prefer sweep, loft, revolve, and helix when the part needs them. Coordinates may use millimeters;
the evaluator will align frames."""


PLAN_TEMPLATES = {"v1": {
    "schema_version": "harnesscad.plan.v1",
    "sample_id": "<provided sample id>",
    "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
    "operations": [
        {
            "id": "base",
            "primitive": "box",
            "combine": "new",
            "center": [0, 0, 0],
            "size": [1.0, 0.5, 0.25],
        }
    ],
}, "v2": {
    "schema_version": "harnesscad.plan.v2",
    "sample_id": "<provided sample id>",
    "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
    "operations": [
        {
            "id": "base",
            "op": "polygon_extrude",
            "combine": "new",
            "workplane": "XY",
            "points": [[-0.5, -0.25], [0.5, -0.25], [0.5, 0.25], [-0.5, 0.25], [-0.5, -0.25]],
            "depth": 0.2,
            "centered": True,
            "offset": [0, 0, 0],
        }
    ],
}, "v3": {
    "schema_version": "harnesscad.plan.v2",
    "sample_id": "<provided sample id>",
    "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
    "operations": [
        {
            "id": "base",
            "op": "polygon_extrude",
            "combine": "new",
            "workplane": "XY",
            "points": [[-0.5, -0.25], [0.5, -0.25], [0.5, 0.25], [-0.5, 0.25], [-0.5, -0.25]],
            "depth": 0.2,
            "centered": True,
            "offset": [0, 0, 0],
        },
        {
            "id": "knob",
            "op": "revolve_profile",
            "combine": "add",
            "workplane": "XZ",
            "profile": [[0.1, 0.02], [0.3, 0.02], [0.3, 0.1], [0.1, 0.1], [0.1, 0.02]],
            "axis": [[0, 0], [1, 0]],
            "angle": 360,
            "offset": [0, 0, 0],
        },
        {
            "id": "rotated_copy",
            "op": "transform",
            "combine": "add",
            "source": "base",
            "rotate": {"origin": [0, 0, 0], "axis": [0, 0, 1], "angle": 90},
        },
    ],
}, "v4": {
    "schema_version": "harnesscad.plan.v3",
    "sample_id": "<provided sample id>",
    "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
    "operations": [
        {
            "id": "tube",
            "op": "sweep_profile",
            "combine": "new",
            "workplane": "XY",
            "profile": [[-0.08, -0.08], [0.08, -0.08], [0.08, 0.08], [-0.08, 0.08], [-0.08, -0.08]],
            "path": [[0, 0, -0.3], [0, 0.1, 0], [0, 0, 0.3]],
            "offset": [0, 0, 0],
        }
    ],
}, "v5": {
    "schema_version": "harnesscad.plan.v3",
    "sample_id": "<provided sample id>",
    "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
    "operations": [
        {
            "id": "base",
            "op": "polygon_extrude",
            "combine": "new",
            "workplane": "XY",
            "points": [[-0.5, -0.25], [0.5, -0.25], [0.5, 0.25], [-0.5, 0.25], [-0.5, -0.25]],
            "depth": 0.08,
            "centered": True,
            "offset": [0, 0, 0],
        },
        {
            "id": "elbow",
            "op": "revolve_profile",
            "combine": "add",
            "workplane": "XY",
            "profile": [[0.22, -0.03], [0.30, -0.03], [0.30, 0.03], [0.22, 0.03], [0.22, -0.03]],
            "axis": [[0, 0], [0, 1]],
            "angle": 90,
            "offset": [0, 0, 0],
        },
        {
            "id": "tube_bend",
            "op": "sweep_profile",
            "combine": "add",
            "workplane": "XY",
            "profile": [[0.22, -0.03], [0.30, -0.03], [0.30, 0.03], [0.22, 0.03], [0.22, -0.03]],
            "path": [[0, 0, 0], [0.2, 0, 0.1], [0.2, 0, 0.3]],
            "offset": [0, 0, 0],
        },
    ],
}, "v6": {
    "schema_version": "harnesscad.plan.v3.1",
    "sample_id": "<provided sample id>",
    "coordinate_system": {"units": "normalized", "origin": [0, 0, 0], "longest_bbox_edge": 1.0},
    "operations": [
        {
            "id": "measured_path",
            "op": "sweep_path_ref",
            "combine": "new",
            "evidence_ref": "path_01",
        }
    ],
}}


def _image_data_url(path: Path, max_edge: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        scale = min(1.0, max_edge / max(image.size))
        if scale < 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_messages(
    row: dict[str, Any],
    condition: str,
    *,
    image_max_edge: int = 1024,
    plan_version: str = "v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if plan_version not in SYSTEM_PROMPTS:
        raise ValueError(f"plan_version 仅支持 {sorted(SYSTEM_PROMPTS)}")
    allowed = modalities_for(condition)
    prompt_sample_id = "s_" + hashlib.sha256(str(row["sample_id"]).encode("utf-8")).hexdigest()[:16]
    content: list[dict[str, Any]] = []
    modality_hashes: dict[str, Any] = {}
    if "images" in allowed:
        for item in row["images"]:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(item["path"]), image_max_edge)}})
        modality_hashes["images"] = [item["sha256"] for item in row["images"]]
    if "point_cloud" in allowed:
        encoded = row["point_cloud"]["encoding"]
        for item in encoded["images"]:
            content.append({"type": "image_url", "image_url": {"url": _image_data_url(Path(item["path"]), image_max_edge)}})
        modality_hashes["point_cloud"] = {
            "source": row["point_cloud"]["sha256"],
            "views": [item["sha256"] for item in encoded["images"]],
            "params": encoded["params"],
        }

    observation = ""
    if "text" in allowed:
        observation = "\nL3 DESCRIPTION:\n" + row["text"]["L3"].strip() + "\n"
        modality_hashes["text"] = sha256_json({"L3": row["text"]["L3"]})
    instruction = (
        f"Create a plan for sample_id {json.dumps(prompt_sample_id)}."
        f"{observation}\nRequired JSON shape (example values are illustrative only):\n"
        + json.dumps(PLAN_TEMPLATES[plan_version], ensure_ascii=False, indent=2)
        + "\nDo not copy the example dimensions. Every operation must include all fields required by the system schema. Output JSON only."
    )
    content.append({"type": "text", "text": instruction})
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[plan_version]}, {"role": "user", "content": content}]
    audit = {
        "allowed_modalities": sorted(allowed),
        "modality_hashes": modality_hashes,
        "prompt_sample_id": prompt_sample_id,
        "plan_version": plan_version,
    }
    return messages, audit


def _extract_json_candidate(text: str) -> str:
    cleaned = text.strip()
    fences = re.findall(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fences:
        cleaned = fences[0].strip()
    start = cleaned.find("{")
    if start < 0:
        return cleaned
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start:index + 1]
    return cleaned[start:]


V1_FIELDS = {"box": {"size"}, "cylinder": {"radius", "height", "axis"}, "sphere": {"radius"}}
V2_REQUIRED_FIELDS = {
    "box": {"combine", "center", "size"},
    "cylinder": {"combine", "center", "radius", "height", "axis"},
    "sphere": {"combine", "center", "radius"},
    "polygon_extrude": {"combine", "workplane", "points", "depth", "centered", "offset"},
    "revolve_profile": {"combine", "workplane", "profile", "axis", "angle", "offset"},
    "hole": {"combine", "workplane", "center", "diameter", "depth"},
    "slot": {"combine", "workplane", "center", "length", "width", "depth", "angle"},
    "transform": {"combine", "source"},
    "fillet": {"radius"},
    "chamfer": {"distance"},
    "linear_pattern": {"combine", "source", "direction", "count", "spacing"},
    "sweep_profile": {"combine", "workplane", "offset"},
    "loft_profiles": {"combine", "workplane", "profiles"},
}
V2_OPTIONAL_FIELDS = {
    "transform": {"translate", "rotate"},
    "fillet": {"edge_axis"},
    "chamfer": {"edge_axis"},
    "polygon_extrude": {"points", "wire"},
    "revolve_profile": {"profile", "wire"},
    "sweep_profile": {"profile", "wire", "path", "helix"},
}

PLAN_V3_PROMPT_VERSIONS = frozenset({"v4", "v5"})
PLAN_V31_PROMPT_VERSIONS = frozenset({"v6"})
PLAN_SCHEMA_PROMPT_VERSIONS = PLAN_V3_PROMPT_VERSIONS | PLAN_V31_PROMPT_VERSIONS
PARSE_PLAN_VERSIONS = frozenset({"v1", "v2", "v3", "v4", "v5", "v6"})
V31_REQUIRED_FIELDS = {
    **V2_REQUIRED_FIELDS,
    "sweep_path_ref": {"combine", "evidence_ref"},
}
V31_OPTIONAL_FIELDS = {
    **V2_OPTIONAL_FIELDS,
    "sweep_profile": V2_OPTIONAL_FIELDS["sweep_profile"] | {"path_wire", "sweep_mode"},
}


def coerce_parse_version(plan_version: str) -> str:
    """把提示词版本映射到 parse/validate 使用的 schema 检查版本。

    prompt v3 仍是 Plan v2；prompt v5 与 v4 同为 Plan v3。
    """
    if plan_version == "v3":
        return "v2"
    return plan_version


def validate_plan(plan: Any, plan_version: str | None = None) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    if not isinstance(plan, dict):
        return [{"path": "$", "code": "not_object"}]
    inferred = {
        "harnesscad.plan.v1": "v1",
        "harnesscad.plan.v2": "v2",
        "harnesscad.plan.v3": "v4",
        "harnesscad.plan.v3.1": "v6",
    }.get(plan.get("schema_version"))
    raw_version = plan_version or inferred
    version = coerce_parse_version(raw_version) if raw_version else raw_version
    if version not in {"v1", "v2", "v4", "v5", "v6"}:
        return [{"path": "$.schema_version", "code": "invalid_schema_version"}]
    allowed_root = {"schema_version", "sample_id", "coordinate_system", "metadata", "operations"}
    for key in sorted(set(plan) - allowed_root):
        issues.append({"path": f"$.{key}", "code": "extra_field"})
    schema_key = "v4" if version in PLAN_V3_PROMPT_VERSIONS else version
    expected_schema = {
        "v1": "harnesscad.plan.v1",
        "v2": "harnesscad.plan.v2",
        "v4": "harnesscad.plan.v3",
        "v6": "harnesscad.plan.v3.1",
    }[schema_key]
    if plan.get("schema_version") != expected_schema:
        issues.append({"path": "$.schema_version", "code": "invalid_schema_version"})
    if not isinstance(plan.get("sample_id"), str):
        issues.append({"path": "$.sample_id", "code": "invalid_sample_id"})
    coordinate = plan.get("coordinate_system")
    if not isinstance(coordinate, dict):
        issues.append({"path": "$.coordinate_system", "code": "invalid_coordinate_system"})
    else:
        if coordinate.get("units") != "normalized" or coordinate.get("origin") != [0, 0, 0] or coordinate.get("longest_bbox_edge") != 1.0:
            issues.append({"path": "$.coordinate_system", "code": "noncanonical_coordinate_system"})
    operations = plan.get("operations")
    max_ops = 128 if version in PLAN_SCHEMA_PROMPT_VERSIONS else 64
    if not isinstance(operations, list) or not 1 <= len(operations) <= max_ops:
        issues.append({"path": "$.operations", "code": "invalid_operations"})
        return issues
    seen = set()
    for index, operation in enumerate(operations):
        path = f"$.operations[{index}]"
        if not isinstance(operation, dict):
            issues.append({"path": path, "code": "not_object"})
            continue
        if version in PLAN_SCHEMA_PROMPT_VERSIONS:
            kind = operation.get("op")
            fields = V31_REQUIRED_FIELDS if version in PLAN_V31_PROMPT_VERSIONS else V2_REQUIRED_FIELDS
            optional = V31_OPTIONAL_FIELDS if version in PLAN_V31_PROMPT_VERSIONS else V2_OPTIONAL_FIELDS
            if kind not in fields:
                issues.append({"path": f"{path}.op", "code": "unsupported_operation"})
                continue
            required = {"id", "op"} | fields[kind]
            allowed = required | optional.get(kind, set())
        elif version == "v1":
            kind = operation.get("primitive")
            if kind not in V1_FIELDS:
                issues.append({"path": f"{path}.primitive", "code": "unsupported_primitive"})
                continue
            required = {"id", "primitive", "combine", "center"} | V1_FIELDS[kind]
            allowed = required
        else:
            kind = operation.get("op")
            if kind not in V2_REQUIRED_FIELDS:
                issues.append({"path": f"{path}.op", "code": "unsupported_operation"})
                continue
            required = {"id", "op"} | V2_REQUIRED_FIELDS[kind]
            allowed = required | V2_OPTIONAL_FIELDS.get(kind, set())
            if kind == "transform" and not ({"translate", "rotate"} & set(operation)):
                issues.append({"path": path, "code": "empty_transform"})
        if not required <= set(operation) or not set(operation) <= allowed:
            issues.append({"path": path, "code": "field_set_mismatch"})
        if not isinstance(operation.get("id"), str) or operation.get("id") in seen:
            issues.append({"path": f"{path}.id", "code": "invalid_or_duplicate_id"})
        seen.add(operation.get("id"))
        combine = operation.get("combine")
        if "combine" in required and (
            combine not in {"new", "add", "cut", "intersect"} or (index == 0) != (combine == "new")
        ):
            issues.append({"path": f"{path}.combine", "code": "invalid_combine"})
        if version in {"v2", "v4", "v5", "v6"} and kind in {"fillet", "chamfer"} and index == 0:
            issues.append({"path": f"{path}.op", "code": "modifier_cannot_be_first"})
        if "center" in required:
            center = operation.get("center")
            if not isinstance(center, list) or len(center) != 3 or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in center
            ):
                issues.append({"path": f"{path}.center", "code": "invalid_vec3"})
    return issues


def _syntax_repair(candidate: str) -> str:
    repaired = candidate.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"(?m)([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)
    return repaired


def _format_repair(plan: dict[str, Any], plan_version: str) -> dict[str, Any]:
    # Removal/coercion only: this repair cannot invent dimensions, centers, primitives, or operations.
    allowed_root = {"schema_version", "sample_id", "coordinate_system", "metadata", "operations"}
    repaired = {key: value for key, value in plan.items() if key in allowed_root}
    operations = repaired.get("operations")
    if isinstance(operations, list):
        clean_ops = []
        for operation in operations:
            if not isinstance(operation, dict):
                clean_ops.append(operation)
                continue
            if plan_version == "v1":
                kind = operation.get("primitive")
                allowed = {"id", "primitive", "combine", "center"} | V1_FIELDS.get(kind, set())
            else:
                kind = operation.get("op")
                fields = V31_REQUIRED_FIELDS if plan_version == "v6" else V2_REQUIRED_FIELDS
                optional = V31_OPTIONAL_FIELDS if plan_version == "v6" else V2_OPTIONAL_FIELDS
                allowed = (
                    {"id", "op"}
                    | fields.get(kind, set())
                    | optional.get(kind, set())
                )
            clean = {key: value for key, value in operation.items() if key in allowed}
            if "id" in clean and isinstance(clean["id"], (int, float)):
                clean["id"] = str(clean["id"])
            clean_ops.append(clean)
        repaired["operations"] = clean_ops
    return repaired


def parse_plan_response(text: str, plan_version: str = "v1") -> dict[str, Any]:
    if plan_version not in PARSE_PLAN_VERSIONS:
        raise ValueError("plan_version 仅支持 v1、v2、v3、v4、v5 或 v6")
    plan_version = coerce_parse_version(plan_version)
    candidate = _extract_json_candidate(text)
    repair: dict[str, Any] | None = None
    try:
        plan = json.loads(candidate)
    except json.JSONDecodeError as first_error:
        repaired_text = _syntax_repair(candidate)
        repair = {"kind": "json_syntax", "before": candidate, "after": repaired_text, "reason": str(first_error)}
        try:
            plan = json.loads(repaired_text)
        except json.JSONDecodeError as second_error:
            return {"ok": False, "plan": None, "issues": [{"path": "$", "code": "invalid_json", "message": str(second_error)}], "repair": repair}
    issues = validate_plan(plan, plan_version)
    if issues and repair is None and isinstance(plan, dict):
        repaired_plan = _format_repair(plan, plan_version)
        if repaired_plan != plan:
            repair = {"kind": "field_format", "before": plan, "after": repaired_plan}
            plan = repaired_plan
            issues = validate_plan(plan, plan_version)
    return {"ok": not issues, "plan": plan, "issues": issues, "repair": repair}
