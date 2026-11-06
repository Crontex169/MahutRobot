#!/usr/bin/env python3
"""
scripts/generate_prism_mesh.py

Milestone 1: Triangular Prism Mesh Generator.
Generates an equilateral triangular prism 3D mesh in standard Wavefront OBJ format.

Specifications:
- Base edge length: a = 35 mm (0.035 m)
- Extrusion height: h = 35 mm (0.035 m)
- Geometric centroid: strictly centered at (0.0, 0.0, 0.0) m (tolerance <= 1e-4 m)
- Watertight 2-manifold topology: 6 shared vertices, 8 triangular faces, 12 edges, Euler characteristic χ = 2
- Exterior counter-clockwise (CCW) face winding with outward normals
- Clean Wavefront OBJ formatting: shared vertices 'v x y z' formatted with fixed-point :.10f
  and face records 'f v1 v2 v3' (1-based indices) without 'vn' to prevent vertex splitting in trimesh 5.0.0
- Built-in multi-tier geometric and SAPIEN 3 validation suite
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import scipy.spatial
import trimesh

# Resolve project root relative to this script: <root>/scripts/generate_prism_mesh.py -> <root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "assets" / "objects" / "triangular_prism.obj"
DEFAULT_BASE_EDGE = 0.035
DEFAULT_HEIGHT = 0.035


def compute_prism_vertices_and_faces(
    base_edge: float = DEFAULT_BASE_EDGE,
    height: float = DEFAULT_HEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes vertices and triangular faces for an equilateral triangular prism centered at (0, 0, 0).

    Geometry formulation:
      Let 'a' = base_edge, 'h' = height.
      Cross section in xy-plane is an equilateral triangle centered at (0, 0):
        - Circumradius: R = a / sqrt(3) (centroid to vertex distance)
        - Inradius:     r = a / (2 * sqrt(3)) (centroid to edge midpoint distance)
        - Altitude:     H = R + r = sqrt(3) / 2 * a

      2D vertices:
        Apex:         (0, +R)
        Bottom-Left:  (-a/2, -r)
        Bottom-Right: (+a/2, -r)
      Centroid in xy:
        x_c = (0 - a/2 + a/2) / 3 = 0.0
        y_c = (R - 2*r) / 3 = (a/sqrt(3) - a/sqrt(3)) / 3 = 0.0

      Extrusion along z-axis:
        Bottom cap at z = -h / 2
        Top cap at z = +h / 2
        Centroid in z: z_c = (-h/2 + h/2) / 2 = 0.0

    Returns:
      vertices: np.ndarray of shape (6, 3), dtype=np.float64
      faces: np.ndarray of shape (8, 3), dtype=np.int64 (0-based indices)
    """
    if base_edge <= 0.0:
        raise ValueError(f"Base edge length must be strictly positive, got {base_edge}")
    if height <= 0.0:
        raise ValueError(f"Height must be strictly positive, got {height}")

    a = float(base_edge)
    h = float(height)

    sqrt3 = math.sqrt(3.0)
    R = a / sqrt3
    r = a / (2.0 * sqrt3)
    half_h = h / 2.0
    half_a = a / 2.0

    # 6 vertices:
    # Indices 0..2: Bottom cap (z = -h/2)
    # Indices 3..5: Top cap (z = +h/2)
    vertices = np.array([
        [0.0, R, -half_h],        # V0: Bottom apex
        [-half_a, -r, -half_h],   # V1: Bottom left
        [half_a, -r, -half_h],    # V2: Bottom right
        [0.0, R, half_h],         # V3: Top apex
        [-half_a, -r, half_h],    # V4: Top left
        [half_a, -r, half_h],     # V5: Top right
    ], dtype=np.float64)

    # 8 triangular faces with outward CCW normals:
    # 1. Bottom cap (pointing down in -Z direction): V0 -> V2 -> V1
    # 2. Top cap (pointing up in +Z direction): V3 -> V4 -> V5
    # 3. Flat front quad (y = -r): split into [1, 2, 5] and [1, 5, 4]
    # 4. Right slant quad: split into [2, 0, 3] and [2, 3, 5]
    # 5. Left slant quad: split into [0, 1, 4] and [0, 4, 3]
    faces = np.array([
        [0, 2, 1],  # Bottom cap (normal: [0, 0, -1])
        [3, 4, 5],  # Top cap (normal: [0, 0, 1])
        [1, 2, 5],  # Side 1 (y = -r, normal: [0, -1, 0])
        [1, 5, 4],  # Side 1
        [2, 0, 3],  # Side 2 (right slant, normal: [sqrt(3)/2, 0.5, 0])
        [2, 3, 5],  # Side 2
        [0, 1, 4],  # Side 3 (left slant, normal: [-sqrt(3)/2, 0.5, 0])
        [0, 4, 3],  # Side 3
    ], dtype=np.int64)

    return vertices, faces


def create_prism_mesh(
    base_edge: float = DEFAULT_BASE_EDGE,
    height: float = DEFAULT_HEIGHT,
) -> trimesh.Trimesh:
    """
    Constructs an in-memory trimesh.Trimesh instance with exact 6 vertices and 8 faces.
    """
    vertices, faces = compute_prism_vertices_and_faces(base_edge=base_edge, height=height)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def save_prism_obj(
    output_path: Union[str, Path] = DEFAULT_OUTPUT_PATH,
    base_edge: float = DEFAULT_BASE_EDGE,
    height: float = DEFAULT_HEIGHT,
) -> str:
    """
    Generates and saves the triangular prism mesh to a standard Wavefront OBJ file.

    Features:
    - Automatically creates parent directories if they do not exist.
    - Uses fixed-point format (:.10f) for vertex coordinates to avoid scientific notation parsing issues.
    - Writes pure shared-vertex indices 'f v1 v2 v3' (1-based) without 'vn' to prevent vertex splitting in trimesh.
    - Idempotent and safe to overwrite existing files.

    Returns:
      Resolved absolute string path to the saved OBJ file.
    """
    dest_path = Path(output_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    vertices, faces = compute_prism_vertices_and_faces(base_edge=base_edge, height=height)

    lines = [
        "# Wavefront OBJ Triangular Prism Mesh",
        f"# Base Edge a = {base_edge:.10f} m, Height h = {height:.10f} m",
        "# Geometric Centroid: (0.0, 0.0, 0.0) m",
        "# Topology: 6 vertices, 8 faces, 12 edges (watertight convex 2-manifold)",
    ]

    for v in vertices:
        lines.append(f"v {v[0]:.10f} {v[1]:.10f} {v[2]:.10f}")

    for f in faces:
        lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")

    lines.append("")  # trailing newline
    content = "\n".join(lines)

    with open(dest_path, "w", encoding="utf-8") as file:
        file.write(content)

    return str(dest_path)


def validate_prism_mesh(
    mesh_or_path: Union[str, Path, trimesh.Trimesh],
    base_edge: float = DEFAULT_BASE_EDGE,
    height: float = DEFAULT_HEIGHT,
    tolerance: float = 1e-4,
    check_sapien: bool = True,
) -> Dict[str, Any]:
    """
    Performs comprehensive verification on a prism mesh or OBJ file.

    Verifications include:
    1. Vertex count == 6 and Face count == 8.
    2. Watertightness (is_watertight == True).
    3. Convexity (is_convex == True and ConvexHull vertex count == 6).
    4. Euler characteristic χ = V - E + F == 2.
    5. Centroid norm <= tolerance (1e-4 m).
    6. Volume matching theoretical formula: V = (sqrt(3)/4) * a^2 * h within 1e-8 m³.
    7. All 8 face normals pointing strictly outward (exterior CCW winding).
    8. Disk round-trip reload preserving clean topology without vertex splitting.
    9. SAPIEN 3 PhysX convex collision and visual component creation (if check_sapien=True).

    Returns:
      Dictionary of verification metrics.
    Raises:
      AssertionError or ValueError if any check fails.
    """
    obj_path: Optional[str] = None
    if isinstance(mesh_or_path, (str, Path)):
        obj_path = str(Path(mesh_or_path).resolve())
        if not os.path.isfile(obj_path):
            raise FileNotFoundError(f"OBJ file not found at: {obj_path}")
        mesh_raw = trimesh.load(obj_path, file_type="obj", process=False)
        mesh_proc = trimesh.load(obj_path, file_type="obj", process=True)
        # Check round-trip preservation
        if len(mesh_raw.vertices) != 6:
            raise AssertionError(
                f"Round-trip reload failed: expected 6 vertices, got {len(mesh_raw.vertices)} (vertex splitting detected)"
            )
        mesh = mesh_raw
    elif isinstance(mesh_or_path, trimesh.Trimesh):
        mesh = mesh_or_path
        mesh_proc = mesh
    else:
        raise TypeError(f"Expected file path or trimesh.Trimesh, got {type(mesh_or_path)}")

    num_vertices = len(mesh.vertices)
    num_faces = len(mesh.faces)
    if num_vertices != 6:
        raise AssertionError(f"Expected 6 vertices, got {num_vertices}")
    if num_faces != 8:
        raise AssertionError(f"Expected 8 faces, got {num_faces}")

    # Watertightness and Euler characteristic
    if not mesh.is_watertight:
        raise AssertionError("Mesh is not watertight (contains open boundaries or non-manifold edges)")
    if mesh.euler_number != 2:
        raise AssertionError(f"Euler characteristic χ must be 2, got {mesh.euler_number}")

    # Convexity
    if not mesh.is_convex:
        raise AssertionError("Mesh is not convex according to trimesh.is_convex")
    hull = scipy.spatial.ConvexHull(mesh.vertices)
    if len(hull.vertices) != 6:
        raise AssertionError(f"ConvexHull has {len(hull.vertices)} vertices, expected 6")

    # Centroid
    centroid = np.asarray(mesh.centroid, dtype=np.float64)
    centroid_norm = float(np.linalg.norm(centroid))
    if centroid_norm > tolerance:
        raise AssertionError(
            f"Centroid norm {centroid_norm:.2e} m exceeds tolerance {tolerance:.2e} m (centroid: {centroid})"
        )

    # Volume
    theoretical_volume = float((math.sqrt(3.0) / 4.0 * (base_edge ** 2)) * height)
    volume_diff = abs(float(mesh.volume) - theoretical_volume)
    if volume_diff > 1e-8:
        raise AssertionError(
            f"Volume diff {volume_diff:.2e} m³ exceeds 1e-8 m³ (mesh: {mesh.volume:.8e}, theoretical: {theoretical_volume:.8e})"
        )

    # Outward normal orientation check
    for idx, face in enumerate(mesh.faces):
        v0 = mesh.vertices[face[0]]
        v1 = mesh.vertices[face[1]]
        v2 = mesh.vertices[face[2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norm_val = np.linalg.norm(cross)
        if norm_val < 1e-12:
            raise AssertionError(f"Face {idx} is degenerate (zero area normal)")
        normal = cross / norm_val
        face_center = (v0 + v1 + v2) / 3.0
        dot = float(np.dot(face_center - centroid, normal))
        if dot <= 0.0:
            raise AssertionError(f"Face {idx} normal does not point outward (dot product: {dot:.4e})")

    # SAPIEN 3 verification
    sapien_verified = False
    if check_sapien:
        try:
            import sapien
            scene = sapien.Scene()
            builder = scene.create_actor_builder()
            mat = sapien.physx.PhysxMaterial(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)

            # If testing from an existing file path, use that; otherwise write temporary file
            temp_file = None
            if obj_path:
                test_path = obj_path
            else:
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
                temp_file.close()
                save_prism_obj(temp_file.name, base_edge=base_edge, height=height)
                test_path = temp_file.name

            try:
                builder.add_convex_collision_from_file(test_path, material=mat)
                builder.add_visual_from_file(test_path)
                actor = builder.build(name="validation_prism")
                comp = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                if comp is None:
                    raise AssertionError("Failed to attach PhysxRigidDynamicComponent in SAPIEN 3")
                shape = comp.collision_shapes[0]
                if shape.vertices.shape != (6, 3):
                    raise AssertionError(f"SAPIEN PhysX shape vertices shape is {shape.vertices.shape}, expected (6, 3)")
                if shape.triangles.shape != (8, 3):
                    raise AssertionError(f"SAPIEN PhysX shape triangles shape is {shape.triangles.shape}, expected (8, 3)")
                cm_norm = float(np.linalg.norm(comp.cmass_local_pose.p))
                if cm_norm > 1e-5:
                    raise AssertionError(f"SAPIEN PhysX center of mass norm {cm_norm:.2e} m exceeds 1e-5 m")
                sapien_verified = True
            finally:
                if temp_file and os.path.exists(temp_file.name):
                    os.remove(temp_file.name)
        except ImportError:
            sapien_verified = False

    return {
        "vertices_count": num_vertices,
        "faces_count": num_faces,
        "is_watertight": bool(mesh.is_watertight),
        "is_convex": bool(mesh.is_convex),
        "euler_number": int(mesh.euler_number),
        "centroid": centroid.tolist(),
        "centroid_norm": centroid_norm,
        "volume": float(mesh.volume),
        "theoretical_volume": theoretical_volume,
        "volume_diff": volume_diff,
        "sapien_verified": sapien_verified,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Equilateral Triangular Prism Mesh Generator for ManiSkill / SAPIEN 3 Simulation"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Target output OBJ file path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--base-edge", "-a",
        type=float,
        default=DEFAULT_BASE_EDGE,
        help=f"Equilateral triangle base edge length in meters (default: {DEFAULT_BASE_EDGE})",
    )
    parser.add_argument(
        "--height", "-H",
        type=float,
        default=DEFAULT_HEIGHT,
        help=f"Prism extrusion height in meters (default: {DEFAULT_HEIGHT})",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip post-generation geometric validation assertions",
    )
    parser.add_argument(
        "--no-sapien",
        action="store_true",
        help="Skip SAPIEN 3 simulation actor validation check",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress informational stdout output",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.quiet:
        print("=" * 80)
        print("🔺 Triangular Prism Mesh Generator (Milestone 1)")
        print("=" * 80)
        print(f"Parameters:")
        print(f"  • Base Edge (a):       {args.base_edge:.6f} m ({args.base_edge * 1000.0:.2f} mm)")
        print(f"  • Height (h):          {args.height:.6f} m ({args.height * 1000.0:.2f} mm)")
        print(f"  • Output Target:       {args.output}")

    try:
        saved_path = save_prism_obj(
            output_path=args.output,
            base_edge=args.base_edge,
            height=args.height,
        )
        if not args.quiet:
            file_size = os.path.getsize(saved_path)
            print(f"\n[OK] Exported Wavefront OBJ to {saved_path} ({file_size} bytes)")

        if not args.no_validate:
            metrics = validate_prism_mesh(
                mesh_or_path=saved_path,
                base_edge=args.base_edge,
                height=args.height,
                tolerance=1e-4,
                check_sapien=not args.no_sapien,
            )
            if not args.quiet:
                print("\nPost-Generation Verification Matrix:")
                print(f"  [PASS] Topology:          {metrics['vertices_count']} vertices, {metrics['faces_count']} faces")
                print(f"  [PASS] Watertightness:    {metrics['is_watertight']} (Euler characteristic χ = {metrics['euler_number']})")
                print(f"  [PASS] Convexity:         {metrics['is_convex']}")
                print(f"  [PASS] Centroid Norm:     {metrics['centroid_norm']:.2e} m (tolerance <= 1.00e-04 m)")
                print(f"  [PASS] Centroid Vector:   [{metrics['centroid'][0]:.6e}, {metrics['centroid'][1]:.6e}, {metrics['centroid'][2]:.6e}]")
                print(f"  [PASS] Volume Match:      {metrics['volume']:.6e} m³ (Delta: {metrics['volume_diff']:.2e} m³)")
                print(f"  [PASS] Outward Normals:   8 / 8 faces verified outward CCW")
                if metrics["sapien_verified"]:
                    print(f"  [PASS] SAPIEN 3 Engine:   Convex collision & visual actor created successfully")

        if not args.quiet:
            print("\n" + "=" * 80)
            print("✅ SUCCESS: Triangular prism mesh generated and validated with 0 errors.")
            print("=" * 80)
        return 0

    except Exception as exc:
        print(f"\n❌ ERROR: Failed to generate or validate prism mesh: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
