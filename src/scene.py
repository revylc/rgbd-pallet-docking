"""Deterministic RGB-D camera observing a pallet made of solid rectangular parts.

World axes: forward, left, up. Camera axes: right, down, forward.
Lengths are metres. Depth is camera Z (not Euclidean range).
"""

from dataclasses import dataclass

import numpy as np


_WIDTH = 640
_HEIGHT = 360
_FOCAL_PX = 560.0
_CAMERA_TO_WORLD = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


@dataclass(frozen=True)
class Pallet:
    x: float = 2.0
    y: float = 0.0
    yaw_deg: float = 12.0
    cargo: str = None


def _pose(pallet):
    yaw = np.deg2rad(pallet.yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.array([pallet.x, pallet.y, 0.25]), rotation


def ground_truth(pallet=None):
    """Evaluator-only labels. Hole order is left-to-right for frontal views."""
    pallet = Pallet() if pallet is None else pallet
    origin, rotation = _pose(pallet)
    holes_local = np.array([[0.0, 0.25, 0.0], [0.0, -0.25, 0.0]])
    return {
        "center_world_m": origin,
        "holes_world_m": holes_local @ rotation.T + origin,
        "inward_world": rotation[:, 0].copy(),
        "yaw_deg": float(pallet.yaw_deg),
    }


def solid_boxes(pallets, occlusion="none"):
    """Scene solids as (origin_world, R_world_local, low, high, RGB)."""
    if occlusion not in {"none", "partial", "severe"}:
        raise ValueError("unknown occlusion: " + str(occlusion))
    pallets = list(pallets)
    boxes = []
    boards = [
        (-0.50, 0.50, 0.10, 0.15, (199, 151, 96)),
        (-0.50, 0.50, -0.15, -0.10, (178, 129, 79)),
        (-0.50, -0.41, -0.25, 0.10, (186, 137, 88)),
        (-0.09, 0.09, -0.25, 0.10, (188, 139, 87)),
        (0.41, 0.50, -0.25, 0.10, (186, 137, 88)),
    ]
    for pallet in pallets:
        origin, rotation = _pose(pallet)
        for y0, y1, z0, z1, colour in boards:
            boxes.append((origin, rotation, (0., y0, z0), (0.40, y1, z1), colour))
        if pallet.cargo is not None:
            colour = {"blue": (49, 107, 210), "red": (207, 57, 67)}[pallet.cargo]
            boxes.append((origin, rotation, (0.025, -0.32, 0.15),
                          (0.37, 0.32, 0.42), colour))
    if pallets and occlusion != "none":
        origin, rotation = _pose(pallets[0])
        # A stationary foreground post blocks the nominal sightline to the
        # left opening. Moving the camera shifts its silhouette by parallax.
        target = origin + (rotation[:, 1] * 0.25 if occlusion == "partial" else 0.0)
        camera = np.array([0., 0., 0.30])
        anchor = camera + 0.52 * (target - camera)
        half_width = 0.078 if occlusion == "partial" else 0.55
        boxes.append((anchor, np.eye(3), (-0.015, -half_width, -0.085),
                      (0.015, half_width, 0.065), (75, 102, 113)))
    return boxes


def _box_entry(origin, rays, low, high):
    """Ray/slab intersection for a camera outside all boxes."""
    entry = np.full(rays.shape[:2], -np.inf)
    exit_ = np.full(rays.shape[:2], np.inf)
    face_axis = np.zeros(rays.shape[:2], dtype=np.int8)
    for axis in range(3):
        direction = rays[..., axis]
        parallel = np.abs(direction) < 1e-12
        divisor = np.where(parallel, 1.0, direction)
        t_low = (low[axis] - origin[axis]) / divisor
        t_high = (high[axis] - origin[axis]) / divisor
        near = np.where(parallel, -np.inf, np.minimum(t_low, t_high))
        far = np.where(parallel, np.inf, np.maximum(t_low, t_high))
        if not low[axis] <= origin[axis] <= high[axis]:
            far[parallel] = -np.inf
        face_axis = np.where(near > entry, axis, face_axis)
        entry = np.maximum(entry, near)
        exit_ = np.minimum(exit_, far)
    return np.where((exit_ >= entry) & (entry > 0), entry, np.inf), face_axis


def render(view_y=0.0):
    """Render the original single-pallet, unobstructed exercise."""
    return render_scene([Pallet()], view_y)


def render_scene(pallets, view_y=0.0, occlusion="none"):
    """RGB uint8, axial depth float32, K, and T_world_camera (4x4).

    view_y translates the level camera along the world-left axis. A fixed
    back wall is at x=4.5 m, so all exercise objects must lie in front of it.
    """
    k = np.array([[_FOCAL_PX, 0., (_WIDTH - 1) / 2],
                  [0., _FOCAL_PX, (_HEIGHT - 1) / 2], [0., 0., 1.]])
    pose = np.eye(4)
    pose[:3, :3] = _CAMERA_TO_WORLD
    pose[:3, 3] = [0.0, float(view_y), 0.30]
    v, u = np.indices((_HEIGHT, _WIDTH), dtype=float)
    optical_rays = np.stack(((u-k[0, 2])/k[0, 0], (v-k[1, 2])/k[1, 1],
                             np.ones_like(u)), axis=-1)
    world_rays = optical_rays @ _CAMERA_TO_WORLD.T
    camera = pose[:3, 3]

    # A vertical back wall and ground plane give every pixel an actual range.
    depth = np.full((_HEIGHT, _WIDTH), 4.5)
    rgb = np.full((_HEIGHT, _WIDTH, 3), (177, 192, 196), dtype=np.uint8)
    floor_t = np.full_like(depth, np.inf)
    np.divide(-camera[2], world_rays[..., 2], out=floor_t,
              where=world_rays[..., 2] < -1e-12)
    floor = floor_t < depth
    depth[floor] = floor_t[floor]
    xy = camera[:2] + floor_t[floor, None] * world_rays[floor, :2]
    tile = (np.floor(xy[:, 0] * 2) + np.floor(xy[:, 1] * 2)) % 2
    rgb[floor] = np.array([119, 137, 141]) + 7 * tile[:, None]

    for origin, rotation, low, high, colour in solid_boxes(pallets, occlusion):
        # Restrict slab tests to the projected AABB, with margin for rounding.
        corners = np.array([[x, y, z] for x in (low[0], high[0])
                            for y in (low[1], high[1]) for z in (low[2], high[2])])
        optical = (corners @ rotation.T + origin - camera) @ _CAMERA_TO_WORLD
        if np.min(optical[:, 2]) <= 1e-8:
            raise ValueError("Scene boxes must lie in front of the camera")
        image = (optical @ k.T)[:, :2] / optical[:, 2, None]
        u0, v0 = np.maximum(np.floor(image.min(axis=0)).astype(int) - 1, 0)
        u1, v1 = np.minimum(np.ceil(image.max(axis=0)).astype(int) + 2, [_WIDTH, _HEIGHT])
        if u1 <= u0 or v1 <= v0:
            continue
        rect = np.s_[v0:v1, u0:u1]
        local_origin = (camera - origin) @ rotation
        local_rays = world_rays[rect] @ rotation
        hit, face = _box_entry(local_origin, local_rays, low, high)
        visible = hit < depth[rect]
        depth[rect][visible] = hit[visible]
        lighting = np.take([1.0, 0.69, 0.84], face[visible])
        rgb[rect][visible] = np.asarray(colour) * lighting[:, None]
    return rgb, depth.astype(np.float32), k, pose
