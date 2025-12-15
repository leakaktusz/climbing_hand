#!/usr/bin/env python3
"""
generate_rock_wall.py

Generate a MuJoCo MJCF XML file containing a rock climbing wall with semi-random holds.

Usage:
    python generate_rock_wall.py            # creates rock_wall.xml
    python generate_rock_wall.py --out wall.xml --cols 6 --rows 8 --prob 0.5 --show

Requirements:
    - dm_control (for mjcf) OR you can still use the generated XML with mujoco.
    - If you pass --show, a working mujoco/mujoco-python install is required.

The generation algorithm (per the report):
  - Divide the wall into tiles of size tile_size x tile_size
  - For each tile, with probability p place a hold.
  - Shift the hold along x by a small random offset (~6% of tile width)
  - Vertical spacing aims to be at least `min_vert_spacing`
"""
import argparse
import random
import math
import os
from pathlib import Path

try:
    from dm_control import mjcf
except Exception:
    mjcf = None

DEFAULT_OUT = "rock_wall.xml"


def build_wall_model(
    width_m=2.0,
    height_m=3.0,
    thickness_m=0.05,
    tile_size=0.25,
    cols=8,
    rows=12,
    place_prob=0.5,
    hold_radius=0.05,
    hold_depth=0.04,
    min_vert_spacing=0.18,
    seed=None,
):
    """Build an MJCF RootElement describing the rock wall and holds."""
    if seed is not None:
        random.seed(seed)

    root = mjcf.RootElement(model="rock_wall")
    from pathlib import Path

    # Add default assets (textures / materials) optionally:
    root.asset.add("texture", name="wood_tex", type="2d", builtin="checker", rgb1=[0.6, 0.45, 0.3], rgb2=[0.55, 0.4, 0.25], width=100, height=100)
    root.asset.add("material", name="wood_mat", texture="wood_tex", reflectance="0.1")
    root.asset.add("mesh", name="hold_1", file=os.path.join(os.path.dirname(__file__), 'meshes', 'Jug_1.STL'))


    world = root.worldbody

    # Add a static wall body centered at x=0,y=0 with the plane facing +x
    wall_body = world.add("body", name="wall", pos=[0.0, 0.0, 0.0])
    # We orient the wall so its surface normal is +x; the geom is a box with size half-extent
    wall_geom = wall_body.add(
        "geom",
        name="wall_geom",
        type="box",
        size=[thickness_m / 2.0, width_m / 2.0, height_m / 2.0],
        pos=[thickness_m / 2.0, 0.0, height_m / 2.0],
        material="wood_mat",
        rgba=[0.7, 0.5, 0.3, 1],
        contype="1",
        conaffinity="1",
    )

    # Decide tile grid: we'll tile across width (y-axis) and height (z-axis)
    # We'll place holds on the surface at x = thickness/2 + hold_depth/2
    tile_w = width_m / cols
    tile_h = height_m / rows

    # Track last placed z per column to attempt spacing
    last_z_in_col = [-1e9] * cols

    holds = []
    for r in range(rows):  # bottom to top
        for c in range(cols):  # left to right (y direction)
            if random.random() > place_prob:
                continue

            # base position of tile center in wall coordinates
            # y runs from -width_m/2 .. +width_m/2
            y_center = (-width_m / 2.0) + (c + 0.5) * tile_w
            z_center = (r + 0.5) * tile_h  # measured from ground (z=0)

            # jitter x (along wall plane horizontal) small percent of tile width (~6%)
            jitter_x = (random.uniform(-0.06, 0.06)) * tile_w

            # jitter y within tile bounds (keep inside wall width)
            jitter_y = random.uniform(-0.2 * tile_w, 0.2 * tile_w)

            # final coordinates (x is slightly out from wall surface)
            x_pos = thickness_m / 2.0 + hold_depth / 2.0 + 0.001  # a hair off the face to avoid exact coplanar
            y_pos = y_center + jitter_y + jitter_x
            z_pos = z_center

            # enforce minimum vertical spacing in this column (approx)
            if z_pos - last_z_in_col[c] < min_vert_spacing:
                # push up enough to satisfy spacing (but keep within wall)
                z_pos = last_z_in_col[c] + min_vert_spacing
                if z_pos > height_m - 0.1:
                    # can't place within wall bounds, skip
                    continue

            last_z_in_col[c] = z_pos

            # Create a small hold as a cylinder standing out from the wall
            hold_name = f"hold_r{r}_c{c}"
            
            mesh_name = f"hold_mesh_{r}_{c}"
            hold_body = world.add("body", name=hold_name, pos=[x_pos, y_pos, z_pos])
            # cylinder axis along y by default? we choose axis along x so it protrudes from wall
            hold_geom = hold_body.add(
                "geom",
                type="mesh",
                mesh="hold_1",
                size=[hold_radius, hold_depth],  # cylinder: size=(radius, length) per mjcf (radius, half-length?) - we keep simple
                pos=[0.0, 0.0, 0.0],
                euler=[0, 0, 0],
                rgba=[0.8, 0.3, 0.2, 1],
                contype="1",
                conaffinity="1",
            )
            holds.append((hold_name, x_pos, y_pos, z_pos))

    return root, holds


def save_xml(root, out_path: Path):
    xml = root.to_xml_string()
    out_path.write_text(xml, encoding="utf-8")
    print(f"Wrote MJCF to: {out_path.resolve()}")


def main():
    ap = argparse.ArgumentParser(description="Generate a MuJoCo rock wall MJCF.")
    ap.add_argument("--out", "-o", default=DEFAULT_OUT, help="Output MJCF filename")
    ap.add_argument("--width", type=float, default=2.0, help="Wall width in meters (horizontal)")
    ap.add_argument("--height", type=float, default=3.0, help="Wall height in meters (vertical)")
    ap.add_argument("--thickness", type=float, default=0.05, help="Wall thickness (m)")
    ap.add_argument("--cols", type=int, default=8, help="Number of tile columns")
    ap.add_argument("--rows", type=int, default=12, help="Number of tile rows")
    ap.add_argument("--prob", type=float, default=0.5, help="Probability of placing a hold per tile")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    ap.add_argument("--show", action="store_true", help="Try to launch mujoco viewer (if installed)")
    ap.add_argument("--tile", type=float, default=None, help="Tile size (overrides cols/width if provided)")
    ap.add_argument("--min-vert-spacing", type=float, default=0.18, help="Minimum vertical spacing between holds in same column")
    args = ap.parse_args()

    # compute tile size if provided
    if args.tile is not None:
        # compute rows/cols from tile if desired; keep supplied rows/cols if set
        pass


    root, holds = build_wall_model(
        width_m=args.width,
        height_m=args.height,
        thickness_m=args.thickness,
        tile_size=None,
        cols=args.cols,
        rows=args.rows,
        place_prob=args.prob,
        hold_radius=0.05,
        hold_depth=0.04,
        min_vert_spacing=args.min_vert_spacing,
        seed=args.seed,
    )

    out_path = Path(args.out)
    save_xml(root, out_path)

    if args.show:
        try:
            import mujoco
            from mujoco import viewer as mj_viewer
            physics = mujoco.MjModel.from_xml_string(root.to_xml_string())
            data = mujoco.MjData(physics)
            print("🎬 Launching MuJoCo viewer (official). Close window to exit.")
            mj_viewer.launch(physics, data)
        except Exception as e:
            print("Could not launch viewer:", e)
            print("You can open the generated XML with mujoco or a compatible viewer.")


if __name__ == "__main__":
    main()
