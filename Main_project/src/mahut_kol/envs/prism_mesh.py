#!/usr/bin/env python3
"""Eşkenar üçgen prizma meshini (a = h = 35 mm, merkez orijinde) OBJ olarak üretir.

6 köşe, 8 üçgen, dışa bakan normaller; `vn` yazılmaz (trimesh köşeleri bölmesin diye).

    python3 scripts/generate_prism_mesh.py [--out assets/objects/triangular_prism.obj] [--base-edge 0.035] [--height 0.035]
"""

import argparse
import math
from pathlib import Path

import numpy as np

DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[3] / "assets" / "objects" / "triangular_prism.obj"

# 0..2 alt kapak, 3..5 üst kapak; her satır dışa bakan CCW üçgen
FACES = np.array([[0, 2, 1], [3, 4, 5], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5], [0, 1, 4], [0, 4, 3]])


def prism_vertices(base_edge: float = 0.035, height: float = 0.035) -> np.ndarray:
    if base_edge <= 0 or height <= 0:
        raise ValueError(f"Boyutlar pozitif olmalı: base_edge={base_edge}, height={height}")
    R, r, a2, h2 = base_edge / math.sqrt(3), base_edge / (2 * math.sqrt(3)), base_edge / 2, height / 2
    tri = [(0.0, R), (-a2, -r), (a2, -r)]  # ağırlık merkezi (0, 0)
    return np.array([(x, y, z) for z in (-h2, h2) for x, y in tri])


def prism_obj_text(base_edge: float = 0.035, height: float = 0.035) -> str:
    lines = [
        "# Wavefront OBJ Triangular Prism Mesh",
        f"# Base Edge a = {base_edge:.10f} m, Height h = {height:.10f} m",
        "# Geometric Centroid: (0.0, 0.0, 0.0) m",
        "# Topology: 6 vertices, 8 faces, 12 edges (watertight convex 2-manifold)",
    ]
    lines += [f"v {x:.10f} {y:.10f} {z:.10f}" for x, y, z in prism_vertices(base_edge, height)]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in FACES]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--base-edge", type=float, default=0.035)
    ap.add_argument("--height", type=float, default=0.035)
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(prism_obj_text(a.base_edge, a.height), encoding="utf-8")
    print(f"Yazıldı: {a.out}")


if __name__ == "__main__":
    main()
