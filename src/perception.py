"""RGB-D pallet entrance baseline for level wooden pallets in the simulation.

The estimator receives observations and camera calibration only. It uses a
wood-colour proposal and fits measured surfaces in 3-D. Metric hole sizes
and spacing are coarse shape priors; no scene pose or segmentation is used.
"""

import cv2
import numpy as np


def _rays(pixels, K):
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    return homogeneous @ np.linalg.inv(K).T


def _fit_face(points, backend):
    """Fit a front-facing vertical plane to candidate cloud in world axes."""
    if len(points) < 100:
        return None
    rng = np.random.default_rng(37)
    chosen = rng.choice(len(points), min(2500, len(points)), replace=False)
    cloud = points[chosen]
    if backend == "open3d":
        import open3d as o3d
        o3d.utility.random.seed(37)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud))
        model, _ = pcd.segment_plane(distance_threshold=0.006, ransac_n=3,
                                     num_iterations=120)
        normal = np.asarray(model[:3])
        if abs(normal[2]) > 0.15 or abs(normal[0]) < 0.65:
            return None
        keep = np.abs(cloud @ normal + model[3]) < 0.006
    else:
        # Samples on the same level vertical face share a horizontal normal.
        pairs = cloud[rng.integers(0, len(cloud), (120, 2))]
        diff = pairs[:, 1, :2] - pairs[:, 0, :2]
        length = np.linalg.norm(diff, axis=1)
        normals = np.column_stack((diff[:, 1], -diff[:, 0])) / np.maximum(length[:, None], 1e-9)
        valid = (length > 0.12) & (np.abs(normals[:, 0]) > 0.65)
        if not valid.any():
            return None
        normals = normals[valid]
        offsets = np.sum(normals * pairs[valid, 0, :2], axis=1)
        distances = cloud[:, :2] @ normals.T - offsets
        inliers = np.abs(distances) < 0.006
        keep = inliers[:, np.argmax(inliers.sum(axis=0))]
    if keep.sum() < 80:
        return None
    face = cloud[keep]
    origin = face.mean(axis=0)
    # Refine the projected line using TLS. The verticality constraint removes
    # pitch/roll from this limited baseline; record its residual for debugging.
    _, _, axes = np.linalg.svd(face[:, :2] - origin[:2], full_matrices=False)
    normal = np.r_[axes[-1], 0.]
    if normal[0] < 0:
        normal *= -1
    return origin, normal


def _intersections(pixels, K, pose, normal, offset):
    world_rays = _rays(pixels, K) @ pose[:3, :3].T
    axial_depth = (offset - pose[:3, 3] @ normal) / (world_rays @ normal)
    return world_rays * axial_depth[:, None] + pose[:3, 3], axial_depth


def _measure(depth, K, pose, selected, bbox, backend):
    x0, y0, x1, y1 = bbox
    result = dict(status="REOBSERVE", reason="insufficient_face_support",
                  bbox_xyxy=bbox, center_world_m=None, holes_world_m=None,
                  inward_world=None, yaw_deg=None, diagnostics={})
    v, u = np.where(selected & np.isfinite(depth) & (depth > 0))
    if not len(u):
        return result
    pixels = np.column_stack((u, v))
    points = (_rays(pixels, K) * depth[v, u, None]) @ pose[:3, :3].T + pose[:3, 3]
    fitted = _fit_face(points, backend)
    if fitted is None:
        return result
    origin, normal = fitted
    residual = (points - origin) @ normal
    front = np.abs(residual) < 0.006
    if front.sum() < 80:
        return result
    offset = origin @ normal
    front_mask = np.zeros((y1-y0, x1-x0), np.uint8)
    front_mask[v[front]-y0, u[front]-x0] = 255
    # Fill isolated noisy pixels without bridging whole posts or missing edges.
    front_mask = cv2.morphologyEx(front_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, tree = cv2.findContours(front_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    diag = result["diagnostics"]
    diag["face_support_points"] = int(front.sum())
    diag["plane_rmse_m"] = float(np.sqrt(np.mean(residual[front]**2)))
    diag["plane_backend"] = backend
    result["reason"] = "two_verified_openings_required"
    tangent = np.array([-normal[1], normal[0], 0.])
    holes = []
    evidence = []
    for idx, contour in enumerate(contours):
        if tree[0, idx, 3] < 0 or cv2.contourArea(contour) < 100:
            continue
        border = contour[:, 0].astype(float) + [x0, y0]
        projected, _ = _intersections(border, K, pose, normal, offset)
        local = np.column_stack((projected @ tangent, projected[:, 2]))
        low, high = local.min(axis=0), local.max(axis=0)
        width, height = high - low
        if not (0.245 < width < 0.40 and 0.155 < height < 0.25):
            continue
        interior = np.zeros_like(front_mask)
        cv2.drawContours(interior, [contour], -1, 255, thickness=-1)
        interior = cv2.erode(interior, np.ones((5, 5), np.uint8))
        iv, iu = np.where(interior)
        samples = np.column_stack((iu[::4]+x0, iv[::4]+y0))
        if not len(samples):
            continue
        _, front_z = _intersections(samples, K, pose, normal, offset)
        measured_z = depth[samples[:, 1], samples[:, 0]]
        # Enough of the aperture must have observed free space beyond the
        # fitted face. A foreground occluder or missing depth is not free.
        free = np.isfinite(measured_z) & (measured_z > front_z + 0.008)
        free_fraction = float(np.mean(free))
        evidence.append(dict(width_m=float(width), height_m=float(height),
                             free_fraction=free_fraction))
        if free_fraction < 0.88:
            continue
        mid = (low + high) / 2
        center = normal * offset + tangent * mid[0] + [0., 0., mid[1]]
        holes.append(center)
    diag["aperture_evidence"] = evidence
    diag["verified_openings"] = len(holes)
    if len(holes) != 2:
        return result
    holes = np.asarray(sorted(holes, key=lambda h: -(h @ tangent)))
    gap = np.linalg.norm(holes[1] - holes[0])
    if not (0.44 < gap < 0.56 and abs(holes[0, 2] - holes[1, 2]) < 0.025):
        result["reason"] = "inconsistent_opening_pair"
        return result
    result.update(status="ACCEPT", reason="two_verified_openings",
                  center_world_m=holes.mean(axis=0).tolist(), holes_world_m=holes.tolist(),
                  inward_world=normal.tolist(),
                  yaw_deg=float(np.degrees(np.arctan2(normal[1], normal[0]))))
    return result


def detect(rgb, depth, K, T_world_camera, backend="numpy"):
    """Find candidates sorted image-left to right. Bboxes use exclusive x1/y1."""
    if backend not in {"numpy", "open3d"}:
        raise ValueError("plane backend must be numpy or open3d")
    colour = np.asarray(rgb, dtype=np.int16)
    # Broad warm wood prior intentionally excludes the red/blue cargo. This
    # simple detector is scoped to wood-coloured pallets, not field deployment.
    r, g, b = np.moveaxis(colour, -1, 0)
    wood = (r - g > 13) & (g - b > 15) & (r - g < 2 * (g - b))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(wood.astype(np.uint8), 8)
    candidates = []
    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < 100 or w < 20 or h < 6:
            continue
        candidates.append(_measure(depth, K, T_world_camera, labels == i,
                                   [x, y, x+w, y+h], backend))
    candidates.sort(key=lambda c: c["bbox_xyxy"][0])
    for index, candidate in enumerate(candidates, start=1):
        candidate["id"] = f"P{index}"
    return candidates
