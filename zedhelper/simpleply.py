from pathlib import Path

import numpy as np

from zedhelper.util import points_isfinite


def write_point_cloud(p: Path, xyzs: np.ndarray, colors: np.ndarray):
    candidate_points = xyzs.shape[0]
    indexes = [i for i in range(candidate_points) if np.isfinite(xyzs[i, :]).all()]

    with open(p, "wt") as f:
        header = f"""ply
format ascii 1.0
comment Created by simpleply
element vertex {len(indexes)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        f.write(header)
        for i in indexes:
            f.write(f"{xyzs[i, 0]:f} {xyzs[i, 1]:f} {xyzs[i, 2]:f} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
