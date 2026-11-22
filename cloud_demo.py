"""Export the observed pallet scene as a coloured PLY and an offline 3-D viewer."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from src.scene import render


def unproject(depth, rgb, K, T, stride):
    if stride < 1:
        raise ValueError("Stride must be positive")
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[v, u]
    valid = np.isfinite(z) & (z > 0)
    pixels = np.column_stack((u[valid], v[valid], np.ones(valid.sum())))
    rays = np.linalg.solve(K, pixels.T).T
    points = (rays * z[valid, None]) @ T[:3, :3].T + T[:3, 3]
    return points.astype(np.float32), rgb[v[valid], u[valid]]


def export(out, stride=4):
    rgb, depth, K, T = render()
    xyz, colors = unproject(depth, rgb, K, T, stride)
    out.mkdir(parents=True, exist_ok=True)
    # PLY stores metres in the fixed world frame and 8-bit sRGB vertex colours.
    record = np.dtype([('xyz','<f4',(3,)), ('rgb','u1',(3,))])
    vertices = np.empty(len(xyz), dtype=record)
    vertices['xyz'], vertices['rgb'] = xyz, colors
    header = ("ply\nformat binary_little_endian 1.0\ncomment world frame: x forward, y left, z up (m)\n"
              f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    (out/'scene.ply').write_bytes(header.encode('ascii') + vertices.tobytes())
    packed = base64.b64encode(vertices.tobytes()).decode('ascii')
    template = Path(__file__).with_name('demos')/'pointcloud.html'
    (out/'pointcloud.html').write_text(template.read_text(encoding='utf-8').replace('__VERTICES__',packed),
                                       encoding='utf-8')
    metadata = {'points':len(vertices), 'stride':stride, 'frame':'world', 'units':'m'}
    (out/'cloud.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(f"{len(vertices):,} coloured 3-D points -> {out}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent/'results'/'cloud')
    parser.add_argument('--stride', type=int, default=4)
    options = parser.parse_args()
    export(options.out, options.stride)
