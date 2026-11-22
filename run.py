"""Replay two RGB-D frames and check a hand-marked geometric baseline."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.geometry import estimate
from src.scene import ground_truth, render


ROOT = Path(__file__).resolve().parent
VIEWS = (("front", 0.0), ("shifted", 0.20))


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda a: a.tolist()), encoding="utf-8")


def save_png(path, bgr):
    ok, data = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError(f"Cannot encode {path}")
    path.write_bytes(data.tobytes())


def pixel(point):
    return tuple(np.rint(point).astype(int))


def caption(image, label):
    bar = np.full((38, image.shape[1], 3), (53, 43, 31), np.uint8)
    cv2.putText(bar, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (250, 250, 250), 1, cv2.LINE_AA)
    return np.vstack((bar, image))


def project(points, K, T):
    camera = (np.asarray(points) - T[:3, 3]) @ T[:3, :3]
    uvw = camera @ K.T
    return uvw[:, :2] / uvw[:, 2:3]


def annotate(image, name):
    points = []
    window = "Mark " + name
    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and y < image.shape[0] and len(points) < 6:
            points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, click)
    try:
        while True:
            frame = image.copy()
            for idx, p in enumerate(points, 1):
                cv2.drawMarker(frame, tuple(p), (0, 255, 255), markerSize=10)
                cv2.putText(frame, str(idx), (p[0]+5, p[1]-3), 0, .45, (0, 255, 255))
            prompt = ("4 spread points on FRONT wood face" if len(points)<4 else
                      "hole centers: image-left then right")
            cv2.imshow(window, np.vstack((frame, caption(frame[:0], f"{len(points)}/6 | {prompt} | Enter: save"))))
            key = cv2.waitKey(40) & 255
            if key in (10, 13) and len(points) == 6:
                return {"plane_pixels": points[:4], "hole_pixels": points[4:]}
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise SystemExit("Annotation cancelled.")
    finally:
        cv2.destroyAllWindows()


def side_view(T, K, holes, samples, result):
    """Orthographic body x-z view; equal scale along both axes."""
    image = np.full((145, 820, 3), 246, np.uint8)
    def xz(point):
        return pixel([35 + 166*point[0], 116 - 166*point[2]])
    camera = xz(T[:3, 3])
    cv2.line(image, (35,116), (810,116), (100,100,100), 1)
    for uv, z, entry in zip(holes, samples, result["centers_world"]):
        raw = T[:3,:3] @ (np.linalg.solve(K, [*uv,1])*z) + T[:3,3]
        cv2.line(image, camera, xz(raw), (180,180,180), 1, cv2.LINE_AA)
        cv2.circle(image, xz(raw), 4, (40,85,230), -1)
        cv2.circle(image, xz(entry), 4, (55,150,50), -1)
    cv2.circle(image, camera, 4, (60,60,60), -1)
    for x in range(5):
        cv2.putText(image, f"{x} m", (28+166*x,137), 0,.4,(60,60,60),1)
    return caption(image, "Body side view (forward / up) | green: entry   orange: raw depth")


def visualize(rgb, depth, K, T, marks, result):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()
    for p in marks["plane_pixels"]:
        cv2.drawMarker(overlay, pixel(p), (0, 255, 255), markerSize=13, thickness=2)
    for p in marks["hole_pixels"]:
        cv2.circle(overlay, pixel(p), 7, (20, 255, 50), 2)
    mid = np.asarray(result["midpoint_world"])
    line = project([mid, mid + .42*np.asarray(result["inward_world"])], K, T)
    cv2.arrowedLine(overlay, pixel(line[0]), pixel(line[1]), (20, 60, 255), 2, tipLength=.35)
    scaled = np.clip((depth-1.)/3., 0., 1.)
    valid = np.isfinite(scaled) & (depth > 0)
    heat = cv2.applyColorMap((np.where(valid, scaled, 0.)*255).astype("uint8"), cv2.COLORMAP_VIRIDIS)
    heat[~valid] = 0
    return np.hstack((caption(overlay, "Front-face samples (+)  |  Hole entries (o)"),
                      caption(heat, "Axial depth  |  dark: <=1 m   yellow: >=4 m")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotate", action="store_true", help="Pick four face points and two hole centers per view")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or ROOT / ("manual_run" if args.annotate else "sample_run")
    out.mkdir(parents=True, exist_ok=True)
    marks = json.loads((ROOT/"annotations.json").read_text(encoding="utf-8"))
    truth = ground_truth()
    rows = []
    midpoints = []
    for name, y in VIEWS:
        folder = out/name
        folder.mkdir(exist_ok=True)
        rgb, depth, K, T = render(y)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        save_png(folder/"rgb.png", bgr)
        np.save(folder/"depth_m.npy", depth)
        save_json(folder/"camera.json", {"K": K, "T_world_camera": T, "depth_unit": "m", "depth_type": "optical_z"})
        pick = annotate(bgr, name) if args.annotate else marks[name]
        save_json(folder/"pixels.json", pick)
        result = estimate(depth, K, T, **pick)
        save_json(folder/"estimate.json", result)
        mid = np.asarray(result["midpoint_world"])
        midpoints.append(mid)
        delta = mid - truth["center_world_m"]
        err = (result["yaw_deg"]-truth["yaw_deg"]+180.) % 360.-180.
        samples = [float(depth[v, u]) for u, v in pick["hole_pixels"]]
        entry_z = ((np.asarray(result["centers_world"])-T[:3,3]) @ T[:3,:3])[:,2]
        row = {"view":name, "midpoint_error_mm": float(np.linalg.norm(delta)*1000),
               "yaw_error_deg": float(abs(err)), "raw_hole_depth_m": samples,
               "estimated_entry_depth_m":entry_z}
        rows.append(row)
        save_png(folder/"side.png", side_view(T, K, pick["hole_pixels"], samples, result))
        save_png(folder/"overlay.png", visualize(rgb, depth, K, T, pick, result))
        print(f"{name}: position error {row['midpoint_error_mm']:.2f} mm | yaw error {abs(err):.4f} deg")
    drift = float(np.linalg.norm(midpoints[1]-midpoints[0])*1000)
    passed = bool(all(r["midpoint_error_mm"] < 10 and r["yaw_error_deg"] < 0.5 for r in rows) and drift < 10)
    save_json(out/"evaluation.json", {"ground_truth":truth, "views":rows, "cross_view_difference_mm":drift,
                                      "criteria":{"midpoint_mm_lt":10,"yaw_deg_lt":0.5,"drift_mm_lt":10},"pass":passed})
    print(f"Cross-view difference: {drift:.2f} mm | {'PASS' if passed else 'REVIEW'} | {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
