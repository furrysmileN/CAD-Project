import cadquery as cq


def build_model(params):
    width = float(params.get("width", 60.0))
    depth = float(params.get("depth", 40.0))
    thickness = float(params.get("thickness", 4.0))
    hole_diameter = float(params.get("hole_diameter", 5.0))
    hole_spacing = float(params.get("hole_spacing", 24.0))

    return (
        cq.Workplane("XY")
        .box(width, depth, thickness)
        .faces(">Z")
        .workplane()
        .pushPoints([(-hole_spacing, 0.0), (hole_spacing, 0.0)])
        .hole(hole_diameter)
    )
