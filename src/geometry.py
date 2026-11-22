"""Recover the entrance pose of a level pallet from annotated RGB-D pixels."""

import numpy as np


def _pixels(values, shape):
    """Return float (u, v) coordinates inside the registered depth image."""
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite pixel coordinates with shape (N, 2)")
    height, width = shape
    if np.any(points < 0) or np.any(points > [width - 1, height - 1]):
        raise ValueError("Annotated pixel lies outside the image")
    return points


def _camera_rays(pixels, intrinsics):
    homogeneous = np.column_stack((pixels, np.ones(len(pixels))))
    return np.linalg.solve(intrinsics, homogeneous.T).T


def estimate(depth, K, T_world_camera, plane_pixels, hole_pixels):
    """Estimate both entrance centers, their midpoint and inward normal.

    depth is registered optical-axis depth in metres. K describes a pinhole
    camera (x right, y down, z forward). T_world_camera maps camera points to
    the world frame (z up). plane_pixels mark solid front-face material.
    hole_pixels mark the projections of the two opening centers. Their depth
    values may lie anywhere behind the pallet and are deliberately unused.

    yaw_deg is the signed angle of the inward normal projected onto the world
    XY plane, measured from +X towards +Y. The returned center order follows
    hole_pixels. plane_rmse_m measures the fit to the annotated face pixels.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("Expected a non-empty, two-dimensional depth image")
    intrinsics = np.asarray(K, dtype=np.float64)
    if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all()
            or not np.allclose(intrinsics[2], [0, 0, 1])
            or not np.allclose(intrinsics[1, 0], 0)
            or min(intrinsics[0, 0], intrinsics[1, 1]) <= 0):
        raise ValueError("Expected pinhole K with positive focal lengths")
    transform = np.asarray(T_world_camera, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1])):
        raise ValueError("Expected a homogeneous 4x4 camera-to-world transform")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1.0)):
        raise ValueError("Camera-to-world rotation must be a proper rotation")

    surface = _pixels(plane_pixels, depth.shape)
    openings = _pixels(hole_pixels, depth.shape)
    if len(surface) < 3 or len(openings) != 2:
        raise ValueError("Mark at least three face pixels and both openings")
    if np.allclose(openings[0], openings[1]):
        raise ValueError("The two openings must have distinct center pixels")

    # Sample the nearest depth pixel and unproject that exact pixel.
    samples = np.rint(surface).astype(int)
    z = depth[samples[:, 1], samples[:, 0]]
    if not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError("Annotated face pixels require finite positive depth")
    face_points = _camera_rays(samples, intrinsics) * z[:, None]
    centroid = face_points.mean(axis=0)
    centered = face_points - centroid
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if singular_values[1] <= max(singular_values[0] * 1e-8, 1e-12):
        raise ValueError("Face points must span a plane; select non-collinear points")
    normal = axes[-1].copy()
    # The camera is at the origin. Select the normal pointing into the pallet.
    if normal @ centroid < 0:
        normal *= -1

    rays = _camera_rays(openings, intrinsics)
    denominators = rays @ normal
    if np.any(np.abs(denominators) < 1e-8):
        raise ValueError("A center ray is parallel to the fitted plane")
    distances = (normal @ centroid) / denominators
    if np.any(distances <= 0):
        raise ValueError("Entrance intersection lies behind the camera")
    centers_camera = rays * distances[:, None]
    centers_world = centers_camera @ rotation.T + transform[:3, 3]
    inward = rotation @ normal
    if np.linalg.norm(inward[:2]) < 1e-8:
        raise ValueError("Fitted front plane has no horizontal entrance direction")
    return {
        "centers_world": centers_world,
        "midpoint_world": centers_world.mean(axis=0),
        "inward_world": inward,
        "yaw_deg": float(np.degrees(np.arctan2(inward[1], inward[0]))),
        "plane_rmse_m": float(np.sqrt(np.mean((centered @ normal) ** 2))),
    }
