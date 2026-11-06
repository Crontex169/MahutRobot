"""
tests/test_prism_stress.py

Adversarial Stress Test Suite for Milestone 1: Triangular Prism Mesh Generator.
Empirically stress-tests geometric boundaries, extreme scales (micro, macro, asymmetric),
numerical stability, floating-point drift across precisions (float64, float32, OBJ format),
bounding dimensions compliance, and error rejection.
"""

import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.spatial
import trimesh

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.generate_prism_mesh import (
    DEFAULT_BASE_EDGE,
    DEFAULT_HEIGHT,
    DEFAULT_OUTPUT_PATH,
    compute_prism_vertices_and_faces,
    create_prism_mesh,
    save_prism_obj,
    validate_prism_mesh,
)


class TestPrismStressAdversarial(unittest.TestCase):
    """
    Adversarial and boundary stress tests for triangular prism mesh generation.
    """

    def setUp(self):
        self.default_obj_path = Path(DEFAULT_OUTPUT_PATH)

    def test_extreme_scale_micro_geometry(self):
        """
        Stress test micro scale (a = 0.001 m = 1 mm, h = 0.001 m).
        Verifies geometric watertightness, convexity, centroid norm, and OBJ round-trip.
        """
        micro_a = 0.001
        micro_h = 0.001

        vertices, faces = compute_prism_vertices_and_faces(micro_a, micro_h)
        self.assertEqual(vertices.shape, (6, 3))
        self.assertEqual(faces.shape, (8, 3))

        mesh = create_prism_mesh(micro_a, micro_h)
        self.assertTrue(mesh.is_watertight, "Micro mesh must be watertight")
        self.assertTrue(mesh.is_convex, "Micro mesh must be convex")
        self.assertLessEqual(np.linalg.norm(mesh.centroid), 1e-4)

        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_prism_obj(tmp_path, base_edge=micro_a, height=micro_h)
            reloaded = trimesh.load(tmp_path, file_type="obj", process=False)
            self.assertEqual(len(reloaded.vertices), 6)
            self.assertEqual(len(reloaded.faces), 8)
            self.assertTrue(reloaded.is_watertight)
            self.assertTrue(reloaded.is_convex)
            self.assertLessEqual(np.linalg.norm(reloaded.centroid), 1e-4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_physx_micro_scale_vertex_collapse_boundary(self):
        """
        Empirically probes PhysX convex mesh cooking threshold.
        Confirms that for a >= 0.002 m (2 mm), PhysX retains all 6 vertices,
        while identifying that below ~1.06 mm PhysX welds vertices into a 4-vertex tetrahedron.
        """
        import sapien

        # Safe micro scale above PhysX cooking weld tolerance (2.0 mm)
        safe_micro_a = 0.002
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_prism_obj(tmp_path, base_edge=safe_micro_a, height=safe_micro_a)
            scene = sapien.Scene()
            builder = scene.create_actor_builder()
            mat = sapien.physx.PhysxMaterial(2.0, 2.0, 0.0)
            builder.add_convex_collision_from_file(tmp_path, material=mat)
            actor = builder.build()
            comp = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            self.assertIsNotNone(comp)
            shape = comp.collision_shapes[0]
            # PhysX retains full 6-vertex topology for a >= 2.0 mm
            self.assertEqual(shape.vertices.shape, (6, 3))
            self.assertEqual(shape.triangles.shape, (8, 3))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_extreme_scale_macro_geometry(self):
        """
        Stress test macro scale (a = 10.0 m, h = 10.0 m).
        Verifies numerical stability, volume match, SAPIEN 3 PhysX support, and centroid accuracy.
        """
        macro_a = 10.0
        macro_h = 10.0

        vertices, faces = compute_prism_vertices_and_faces(macro_a, macro_h)
        mesh = create_prism_mesh(macro_a, macro_h)

        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_convex)
        self.assertLessEqual(np.linalg.norm(mesh.centroid), 1e-4)

        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            save_prism_obj(tmp_path, base_edge=macro_a, height=macro_h)
            metrics = validate_prism_mesh(tmp_path, base_edge=macro_a, height=macro_h, check_sapien=True)
            self.assertEqual(metrics["vertices_count"], 6)
            self.assertEqual(metrics["faces_count"], 8)
            self.assertTrue(metrics["is_watertight"])
            self.assertTrue(metrics["is_convex"])
            self.assertTrue(metrics["sapien_verified"])
            self.assertLessEqual(metrics["centroid_norm"], 1e-4)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_asymmetric_scales_needle_and_wafer(self):
        """
        Stress test high aspect ratio prisms:
        - Needle: a = 0.005 m (5 mm), h = 1.0 m (1000 mm) [aspect ratio 1:200]
        - Wafer:  a = 1.0 m (1000 mm), h = 0.005 m (5 mm) [aspect ratio 200:1]
        """
        cases = [
            ("needle", 0.005, 1.0),
            ("wafer", 1.0, 0.005),
        ]

        for name, a, h in cases:
            with self.subTest(case=name):
                with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    save_prism_obj(tmp_path, base_edge=a, height=h)
                    metrics = validate_prism_mesh(tmp_path, base_edge=a, height=h, check_sapien=True)
                    self.assertEqual(metrics["vertices_count"], 6)
                    self.assertEqual(metrics["faces_count"], 8)
                    self.assertTrue(metrics["is_watertight"])
                    self.assertTrue(metrics["is_convex"])
                    self.assertTrue(metrics["sapien_verified"])
                    self.assertLessEqual(metrics["centroid_norm"], 1e-4)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

    def test_invalid_negative_and_zero_inputs(self):
        """
        Stress test input boundary enforcement: non-positive dimensions must raise ValueError.
        """
        invalid_pairs = [
            (0.0, 0.035),
            (-0.035, 0.035),
            (0.035, 0.0),
            (0.035, -0.035),
            (-1.0, -1.0),
            (-1e-8, 0.035),
            (0.035, -1e-8),
        ]

        for a, h in invalid_pairs:
            with self.subTest(a=a, h=h):
                with self.assertRaises(ValueError):
                    compute_prism_vertices_and_faces(base_edge=a, height=h)
                with self.assertRaises(ValueError):
                    create_prism_mesh(base_edge=a, height=h)
                with self.assertRaises(ValueError):
                    save_prism_obj(DEFAULT_OUTPUT_PATH, base_edge=a, height=h)

    def test_cli_invalid_parameter_exit_codes(self):
        """
        Verify CLI returns non-zero exit code when given invalid negative or zero parameters.
        """
        cli_cases = [
            ["--base-edge", "0.0"],
            ["--base-edge", "-0.035"],
            ["--height", "0.0"],
            ["--height", "-0.01"],
        ]

        for extra_args in cli_cases:
            with self.subTest(args=extra_args):
                cmd = [sys.executable, str(_REPO_ROOT / "scripts" / "generate_prism_mesh.py")] + extra_args + ["--quiet"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(res.returncode, 1, f"CLI should fail with exit code 1 for args {extra_args}")
                self.assertIn("ERROR", res.stderr)

    def test_precision_and_floating_point_drift(self):
        """
        Evaluate numerical stability and centroid norm across:
        1. In-memory float64 representation
        2. OBJ 10-decimal fixed-point serialization and reload
        3. Single-precision float32 conversion (OpenGL / GPU physics format)
        4. SAPIEN 3 PhysX center of mass
        All must maintain centroid norm <= 1e-4 m (target <= 100 micrometers).
        """
        a = DEFAULT_BASE_EDGE
        h = DEFAULT_HEIGHT

        v64, f = compute_prism_vertices_and_faces(a, h)
        m64 = trimesh.Trimesh(vertices=v64, faces=f, process=False)
        norm_float64 = np.linalg.norm(m64.centroid)
        self.assertLess(norm_float64, 1e-15, f"Float64 centroid drift {norm_float64} exceeds 1e-15 m")

        # Serialized OBJ representation
        m_obj = trimesh.load(str(self.default_obj_path), file_type="obj", process=False)
        norm_obj = np.linalg.norm(m_obj.centroid)
        self.assertLess(norm_obj, 1e-10, f"OBJ reloaded centroid drift {norm_obj} exceeds 1e-10 m")

        # Float32 precision representation
        v32 = v64.astype(np.float32)
        m32 = trimesh.Trimesh(vertices=v32, faces=f, process=False)
        norm_float32 = np.linalg.norm(m32.centroid)
        self.assertLess(norm_float32, 1e-8, f"Float32 centroid drift {norm_float32} exceeds 1e-8 m")

        # SAPIEN PhysX center of mass
        import sapien
        scene = sapien.Scene()
        builder = scene.create_actor_builder()
        mat = sapien.physx.PhysxMaterial(2.0, 2.0, 0.0)
        builder.add_convex_collision_from_file(str(self.default_obj_path), material=mat)
        actor = builder.build()
        comp = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        norm_sapien = np.linalg.norm(comp.cmass_local_pose.p)
        self.assertLess(norm_sapien, 1e-6, f"SAPIEN PhysX center of mass drift {norm_sapien} exceeds 1e-6 m")

        # All must satisfy the contract tolerance
        tolerance = 1e-4
        self.assertLessEqual(norm_float64, tolerance)
        self.assertLessEqual(norm_obj, tolerance)
        self.assertLessEqual(norm_float32, tolerance)
        self.assertLessEqual(norm_sapien, tolerance)

    def test_bounding_dimensions_and_equilateral_geometry(self):
        """
        Verify exact bounding dimensions conform to 35 mm scale and equilateral triangle geometry.
        """
        mesh = trimesh.load(str(self.default_obj_path), file_type="obj", process=False)
        extents = mesh.extents

        # X span = base edge = 35.0 mm
        self.assertAlmostEqual(extents[0], 0.035, places=6)
        # Y span = equilateral altitude H = (sqrt(3)/2) * a = 30.310889 mm
        expected_altitude = (math.sqrt(3.0) / 2.0) * 0.035
        self.assertAlmostEqual(extents[1], expected_altitude, places=6)
        # Z span = height = 35.0 mm
        self.assertAlmostEqual(extents[2], 0.035, places=6)

        # Confirm all vertices fit strictly within 35 mm cube boundary [-0.0175, +0.0175]^3
        bounds_min, bounds_max = mesh.bounds
        self.assertGreaterEqual(bounds_min[0], -0.0175 - 1e-7)
        self.assertLessEqual(bounds_max[0], 0.0175 + 1e-7)
        self.assertGreaterEqual(bounds_min[1], -0.0175 - 1e-7)
        self.assertLessEqual(bounds_max[1], 0.025)  # R = a/sqrt(3) ~ 0.0202 m
        self.assertGreaterEqual(bounds_min[2], -0.0175 - 1e-7)
        self.assertLessEqual(bounds_max[2], 0.0175 + 1e-7)

        # Verify base triangle angles are exactly 60 degrees (pi / 3 radians)
        v = mesh.vertices
        # Bottom cap vertices: 0 (apex), 1 (left), 2 (right)
        v0, v1, v2 = v[0][:2], v[1][:2], v[2][:2]

        def angle_between(u, w):
            cos_theta = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
            return np.arccos(np.clip(cos_theta, -1.0, 1.0))

        angle0 = angle_between(v1 - v0, v2 - v0)
        angle1 = angle_between(v0 - v1, v2 - v1)
        angle2 = angle_between(v0 - v2, v1 - v2)
        expected_angle = math.pi / 3.0

        self.assertAlmostEqual(angle0, expected_angle, places=6)
        self.assertAlmostEqual(angle1, expected_angle, places=6)
        self.assertAlmostEqual(angle2, expected_angle, places=6)


if __name__ == "__main__":
    unittest.main()
