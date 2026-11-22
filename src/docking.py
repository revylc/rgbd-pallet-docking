"""Continuous swept-volume test for two level, fixed-yaw, translating tines.

All coordinates are in metres in a world frame with positive z up. Both
target geometry and obstacle poses belong to the independent evaluator.
"""

import numpy as np


TINE_LENGTH_M = 0.55
TINE_WIDTH_M = 0.16
TINE_HEIGHT_M = 0.06
HALF_SPACING_M = 0.25
START_GAP_M = 0.22
_EPS = 1e-10


def _basis(yaw_deg):
    angle = np.deg2rad(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def _world_pose(record):
    point = record.get("midpoint_world", record.get("center_world_m"))
    point = np.asarray(point, dtype=float)
    yaw = float(record["yaw_deg"])
    if point.shape != (3,) or not np.isfinite(point).all() or not np.isfinite(yaw):
        raise ValueError("Pose requires a finite world XYZ point and yaw_deg")
    return point, _basis(yaw)


def _pallet_solids(origin, axes):
    # Same two openings and five solids as the standalone lesson-1 renderer.
    # Per-run truth may also supply full scene solids via obstacles.
    bounds = [
        ("top deck", -.50, .50, .10, .15),
        ("bottom deck", -.50, .50, -.15, -.10),
        ("right post", -.50, -.41, -.25, .10),
        ("center post", -.09, .09, -.25, .10),
        ("left post", .41, .50, -.25, .10),
    ]
    for label, y0, y1, z0, z1 in bounds:
        xy = origin[:2] + axes @ [.20, (y0 + y1) / 2]
        yield label, xy, axes, np.array([.20, (y1-y0)/2]), origin[2]+z0, origin[2]+z1


def _other_solids(obstacles):
    for i, box in enumerate(obstacles):
        origin, rotation, low, high = (np.asarray(v, dtype=float) for v in box[:4])
        if (origin.shape != (3,) or rotation.shape != (3, 3)
                or low.shape != (3,) or high.shape != (3,) or np.any(high <= low)
                or not all(np.isfinite(a).all() for a in (low, high, rotation, origin))
                or not np.allclose(rotation[:, 2], [0, 0, 1])
                or not np.allclose(rotation.T @ rotation, np.eye(3))):
            raise ValueError("Obstacle must be a finite level box with positive extents")
        center = origin + rotation @ ((low+high)/2)
        half = (high-low)/2
        yield f"scene box {i}", center[:2], rotation[:2, :2], half[:2], center[2]-half[2], center[2]+half[2]


def _first_contact(center, axes, half_size, movement, solid):
    """Intersect the four SAT-axis overlap time intervals over t in [0, 1]."""
    _, other_center, other_axes, other_half, _, _ = solid
    delta = other_center - center
    enter, leave = 0., 1.
    for axis in (axes[:, 0], axes[:, 1], other_axes[:, 0], other_axes[:, 1]):
        radius = np.abs(axis @ axes) @ half_size + np.abs(axis @ other_axes) @ other_half
        offset = float(axis @ delta)
        speed = float(axis @ movement)
        if abs(speed) < _EPS:
            if abs(offset) > radius + _EPS:
                return None
            continue
        a, b = (offset-radius)/speed, (offset+radius)/speed
        enter, leave = max(enter, min(a,b)), min(leave, max(a,b))
        if enter > leave + _EPS:
            return None
    return float(np.clip(enter, 0., 1.))


def tine_corners(estimated_pose, tip_travel_m):
    """Two world-XY tine polygons; first two vertices are their leading edges.

    tip_travel_m denotes tip position relative to the commanded entrance.
    At -START_GAP_M both tines are in the pre-dock pose.
    """
    center, axes = _world_pose(estimated_pose)
    fore = np.array([tip_travel_m, tip_travel_m, tip_travel_m-TINE_LENGTH_M,
                     tip_travel_m-TINE_LENGTH_M])
    across = TINE_WIDTH_M/2 * np.array([-1, 1, 1, -1])
    return [center[:2] + np.column_stack((fore, across + side)) @ axes.T
            for side in (HALF_SPACING_M, -HALF_SPACING_M)]


def validate(pallet_pose, estimated_pose, obstacles=(), *, insertion_m=.34,
             required_depth_m=.30):
    """Test a straight insertion against a truth pallet and optional scene boxes.

    A successful motion starts in front of the true entrance, directs one
    physical tine through each opening, stays contact-free for the entire
    trajectory, and inserts both leading edges at least required_depth_m.
    obstacles accepts scene.solid_boxes tuples (origin, R, low, high, color).
    """
    if not (np.isfinite(insertion_m) and np.isfinite(required_depth_m)
            and insertion_m >= 0 and 0 < required_depth_m <= .40):
        raise ValueError("Insertion must be nonnegative and required depth in (0, .40]")
    target, target_axes = _world_pose(pallet_pose)
    command, axes = _world_pose(estimated_pose)
    start = tine_corners(estimated_pose, -START_GAP_M)
    delta = axes[:, 0] * (START_GAP_M + insertion_m)
    z0, z1 = command[2]-TINE_HEIGHT_M/2, command[2]+TINE_HEIGHT_M/2
    solids = list(_pallet_solids(target, target_axes)) + list(_other_solids(obstacles))
    contact, label = None, None
    if z0 <= 0:
        contact, label = 0., "floor"
    for index, corners in enumerate(start):
        for solid in solids:
            if z1 < solid[4]-_EPS or z0 > solid[5]+_EPS:
                continue
            hit = _first_contact(corners.mean(axis=0), axes,
                                 np.array([TINE_LENGTH_M, TINE_WIDTH_M])/2, delta, solid)
            if hit is not None and (contact is None or hit < contact):
                contact, label = hit, f"tine {index+1} / {solid[0]}"

    start_edges = np.array([corners[:2] for corners in start])
    start_local = (start_edges-target[:2]) @ target_axes
    local_delta = delta @ target_axes
    fraction = 1. if contact is None else contact
    end_local = start_local + fraction*local_delta
    depth = max(0., float(end_local[..., 0].min()))
    status, reason = "success", "Both tines entered their openings to the required depth"
    if contact is not None:
        status, reason = "collision", label
    elif np.max(start_local[..., 0]) > _EPS or local_delta[0] <= _EPS:
        status, reason = "missed_opening", "Start tips must approach from in front of the target"
    else:
        # At the entrance plane intersect each moving tip-corner line with x=0.
        ingress_y = start_local[..., 1] - start_local[..., 0]*local_delta[1]/local_delta[0]
        offsets = ingress_y - np.array([[HALF_SPACING_M], [-HALF_SPACING_M]])
        if np.any(np.abs(offsets) >= .16-_EPS) or z0 <= target[2]-.10 or z1 >= target[2]+.10:
            status, reason = "missed_opening", "Tip paths do not pass through the two openings"
        elif depth + _EPS < required_depth_m:
            status, reason = "insufficient_depth", "At least one tip fell short"
    if status == "missed_opening":
        depth = 0.
    return {
        "success": status == "success", "status": status, "reason": reason,
        "achieved_depth_m": depth, "required_depth_m": float(required_depth_m),
        "contact_fraction": contact, "travel_fraction": float(fraction),
    }
