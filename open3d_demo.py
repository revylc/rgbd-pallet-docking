"""Fit pallet faces in three RGB-D observations using a selected plane backend.

With Open3D installed, also load a world-frame colour PLY into an Open3D
PointCloud. Pass --show to inspect it interactively after the measurements.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.perception import detect
from src.scene import Pallet, render_scene


ROOT = Path(__file__).resolve().parent


def measure(backend):
    cases = [
        ("clear", "none", 0.0),
        ("partial_before_move", "partial", 0.0),
        ("partial_after_move", "partial", -0.5),
    ]
    results = []
    for label, occlusion, camera_y in cases:
        observation = render_scene([Pallet()], view_y=camera_y, occlusion=occlusion)
        candidates = detect(*observation, backend=backend)
        results.append({
            "case": label, "camera_y_m": camera_y, "candidates": candidates
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "open3d"], default="numpy")
    parser.add_argument("--ply", type=Path, default=ROOT / "results" / "cloud" / "scene.ply",
                        help="Open3D inspection cloud, separately from RGB-D estimation")
    parser.add_argument("--show", action="store_true",
                        help="Open the loaded PLY in Open3D's interactive viewer")
    args = parser.parse_args()
    if args.show and args.backend != "open3d":
        parser.error("--show requires --backend open3d")
    output = {"backend": args.backend}
    if args.backend == "open3d":
        try:
            import open3d as o3d
        except ModuleNotFoundError as exc:
            if exc.name != "open3d":
                raise
            parser.exit(2, "Install optional open3d in this Python environment first.\n")
        if not args.ply.is_file():
            parser.error(f"Missing PLY: {args.ply}. Run python cloud_demo.py first.")
        cloud = o3d.io.read_point_cloud(str(args.ply))
        if cloud.is_empty() or not cloud.has_colors():
            parser.error("Expected a nonempty coloured world-frame PLY")
        output["open3d_version"] = o3d.__version__
        output["inspection_cloud"] = {
            "points": len(cloud.points),
            "extent_m": np.asarray(cloud.get_axis_aligned_bounding_box().get_extent()).tolist()
        }
    output["observations"] = measure(args.backend)
    print(json.dumps(output, indent=2, allow_nan=False))
    if args.show:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        o3d.visualization.draw_geometries([cloud, axes], window_name="World-frame pallet RGB-D")


if __name__ == "__main__":
    main()
