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

# Add repo root to sys.path so scripts and modules can be imported
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


class TestPrismMesh(unittest.TestCase):
    """
    Comprehensive test suite for Milestone 1: Triangular Prism Mesh Generator.
    Verifies mathematical geometry, Wavefront OBJ format, topological watertightness,
    convexity, outward CCW normals, SAPIEN 3 PhysX integration, and CLI behavior.
    """

    def setUp(self):
        self.default_obj_path = Path(DEFAULT_OUTPUT_PATH)
        self.a = DEFAULT_BASE_EDGE
        self.h = DEFAULT_HEIGHT

    def test_compute_prism_vertices_and_faces_default(self):
        """Verify vertex and face arrays generated for default 35mm dimensions."""
        vertices, faces = compute_prism_vertices_and_faces(self.a, self.h)

        self.assertEqual(vertices.shape, (6, 3))
        self.assertEqual(faces.shape, (8, 3))
        self.assertEqual(vertices.dtype, np.float64)
        self.assertEqual(faces.dtype, np.int64)

        # Check vertex bounding box
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)

        # X span: [-a/2, +a/2] -> width = a
        self.assertAlmostEqual(max_coords[0] - min_coords[0], self.a, places=7)
        # Z span: [-h/2, +h/2] -> height = h
        self.assertAlmostEqual(max_coords[2] - min_coords[2], self.h, places=7)

        # Bottom cap edge lengths should all equal 'a'
        e01 = np.linalg.norm(vertices[1] - vertices[0])
        e12 = np.linalg.norm(vertices[2] - vertices[1])
        e20 = np.linalg.norm(vertices[0] - vertices[2])
        self.assertAlmostEqual(e01, self.a, places=7)
        self.assertAlmostEqual(e12, self.a, places=7)
        self.assertAlmostEqual(e20, self.a, places=7)

        # Vertical edge lengths should all equal 'h'
        for i in range(3):
            v_len = np.linalg.norm(vertices[i + 3] - vertices[i])
            self.assertAlmostEqual(v_len, self.h, places=7)

    def test_geometric_invariants_and_centroid(self):
        """Verify strict centroid centering at (0, 0, 0) and theoretical volume match."""
        mesh = create_prism_mesh(self.a, self.h)

        # Centroid tolerance: requirement specifies <= 1e-4 m
        centroid_norm = np.linalg.norm(mesh.centroid)
        self.assertLessEqual(
            centroid_norm,
            1e-4,
            f"Centroid norm {centroid_norm:.2e} m exceeds tolerance 1e-4 m",
        )

        # Analytical volume: (sqrt(3)/4) * a^2 * h
        expected_volume = (math.sqrt(3.0) / 4.0 * (self.a ** 2)) * self.h
        self.assertAlmostEqual(mesh.volume, expected_volume, places=8)

        # Watertightness and Euler characteristic (χ = V - E + F = 6 - 12 + 8 = 2)
        self.assertTrue(mesh.is_watertight, "Prism mesh must be watertight")
        self.assertEqual(mesh.euler_number, 2, "Euler characteristic must be 2")

        # Convexity
        self.assertTrue(mesh.is_convex, "Prism mesh must be convex")
        hull = scipy.spatial.ConvexHull(mesh.vertices)
        self.assertEqual(len(hull.vertices), 6, "Convex hull must include all 6 vertices")
        self.assertAlmostEqual(hull.volume, expected_volume, places=8)

    def test_outward_normals_and_ccw_winding(self):
        """Verify all 8 face normals point strictly outward from the geometric centroid."""
        vertices, faces = compute_prism_vertices_and_faces(self.a, self.h)
        centroid = vertices.mean(axis=0)

        for i, face in enumerate(faces):
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            cross = np.cross(v1 - v0, v2 - v0)
            normal = cross / np.linalg.norm(cross)
            face_center = (v0 + v1 + v2) / 3.0
            dot_product = np.dot(face_center - centroid, normal)
            self.assertGreater(
                dot_product,
                0.0,
                f"Face {i} ({face}) normal points inward: dot product {dot_product}",
            )

        # Verify bottom cap normal is [0, 0, -1] and top cap normal is [0, 0, 1]
        n_bottom = np.cross(vertices[faces[0][1]] - vertices[faces[0][0]], vertices[faces[0][2]] - vertices[faces[0][0]])
        n_bottom /= np.linalg.norm(n_bottom)
        np.testing.assert_allclose(n_bottom, [0.0, 0.0, -1.0], atol=1e-7)

        n_top = np.cross(vertices[faces[1][1]] - vertices[faces[1][0]], vertices[faces[1][2]] - vertices[faces[1][0]])
        n_top /= np.linalg.norm(n_top)
        np.testing.assert_allclose(n_top, [0.0, 0.0, 1.0], atol=1e-7)

    def test_invalid_dimensions_rejection(self):
        """Verify ValueError is raised for non-positive geometric parameters."""
        with self.assertRaises(ValueError):
            compute_prism_vertices_and_faces(base_edge=0.0, height=0.035)

        with self.assertRaises(ValueError):
            compute_prism_vertices_and_faces(base_edge=-0.035, height=0.035)

        with self.assertRaises(ValueError):
            compute_prism_vertices_and_faces(base_edge=0.035, height=0.0)

        with self.assertRaises(ValueError):
            compute_prism_vertices_and_faces(base_edge=0.035, height=-0.01)

    def test_parameterized_custom_dimensions(self):
        """Verify generator supports custom edge lengths and heights."""
        custom_a = 0.050
        custom_h = 0.020
        mesh = create_prism_mesh(custom_a, custom_h)

        self.assertEqual(len(mesh.vertices), 6)
        self.assertEqual(len(mesh.faces), 8)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_convex)

        expected_vol = (math.sqrt(3.0) / 4.0 * (custom_a ** 2)) * custom_h
        self.assertAlmostEqual(mesh.volume, expected_vol, places=8)
        self.assertLessEqual(np.linalg.norm(mesh.centroid), 1e-4)

    def test_obj_file_on_disk(self):
        """Verify the generated assets/objects/triangular_prism.obj file on disk."""
        self.assertTrue(
            self.default_obj_path.is_file(),
            f"Expected OBJ file does not exist at {self.default_obj_path}",
        )

        with open(self.default_obj_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        v_lines = [l.strip() for l in lines if l.startswith("v ")]
        f_lines = [l.strip() for l in lines if l.startswith("f ")]
        vn_lines = [l.strip() for l in lines if l.startswith("vn ")]
        vt_lines = [l.strip() for l in lines if l.startswith("vt ")]

        self.assertEqual(len(v_lines), 6, "OBJ must have exactly 6 vertex records")
        self.assertEqual(len(f_lines), 8, "OBJ must have exactly 8 face records")
        self.assertEqual(len(vn_lines), 0, "OBJ must omit 'vn' to prevent vertex splitting in trimesh 5.0.0")
        self.assertEqual(len(vt_lines), 0, "OBJ must omit 'vt' texture coordinates")

        # Check 1-based indexing in face records
        all_indices = []
        for fl in f_lines:
            tokens = fl.split()[1:]
            self.assertEqual(len(tokens), 3, "Each face must be triangular (3 vertices)")
            for t in tokens:
                idx = int(t)
                all_indices.append(idx)
                self.assertGreaterEqual(idx, 1)
                self.assertLessEqual(idx, 6)

        self.assertEqual(set(all_indices), {1, 2, 3, 4, 5, 6})

    def test_round_trip_trimesh_reload_preserves_topology(self):
        """Verify loading the exported OBJ with process=False and process=True avoids vertex splitting."""
        # Raw reload without processing
        m_raw = trimesh.load(str(self.default_obj_path), file_type="obj", process=False)
        self.assertEqual(len(m_raw.vertices), 6, "process=False must preserve 6 vertices")
        self.assertEqual(len(m_raw.faces), 8)
        self.assertTrue(m_raw.is_watertight)
        self.assertTrue(m_raw.is_convex)
        self.assertLessEqual(np.linalg.norm(m_raw.centroid), 1e-4)

        # Standard processing reload
        m_proc = trimesh.load(str(self.default_obj_path), file_type="obj", process=True)
        self.assertEqual(len(m_proc.vertices), 6, "process=True must preserve 6 vertices")
        self.assertEqual(len(m_proc.faces), 8)
        self.assertTrue(m_proc.is_watertight)
        self.assertTrue(m_proc.is_convex)

    def test_save_prism_obj_directory_auto_creation(self):
        """Verify save_prism_obj automatically creates parent directories when nested."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nested_path = Path(tmp_dir) / "sub1" / "sub2" / "custom_prism.obj"
            self.assertFalse(nested_path.parent.exists())

            saved = save_prism_obj(nested_path, base_edge=0.035, height=0.035)
            self.assertTrue(os.path.isfile(saved))
            self.assertTrue(nested_path.is_file())

            # Validate exported file
            metrics = validate_prism_mesh(saved, base_edge=0.035, height=0.035, check_sapien=False)
            self.assertTrue(metrics["is_watertight"])
            self.assertTrue(metrics["is_convex"])
            self.assertEqual(metrics["vertices_count"], 6)

    def test_validate_prism_mesh_success_and_failure(self):
        """Verify validate_prism_mesh returns metric dictionary and raises on invalid mesh."""
        metrics = validate_prism_mesh(self.default_obj_path, base_edge=self.a, height=self.h)
        self.assertEqual(metrics["vertices_count"], 6)
        self.assertEqual(metrics["faces_count"], 8)
        self.assertTrue(metrics["is_watertight"])
        self.assertTrue(metrics["is_convex"])
        self.assertTrue(metrics["sapien_verified"])

        # Intolerant tolerance should raise AssertionError
        with self.assertRaises(AssertionError):
            validate_prism_mesh(self.default_obj_path, tolerance=1e-15)

    def test_sapien_actor_building(self):
        """Verify SAPIEN 3 creates dynamic actor with PhysX collision and visual component."""
        import sapien

        scene = sapien.Scene()
        builder = scene.create_actor_builder()
        mat = sapien.physx.PhysxMaterial(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)

        builder.add_convex_collision_from_file(str(self.default_obj_path), material=mat)
        render_mat = sapien.render.RenderMaterial()
        render_mat.base_color = [0.85, 0.15, 0.15, 1.0]
        builder.add_visual_from_file(str(self.default_obj_path), material=render_mat)

        actor = builder.build(name="test_prism_actor")
        self.assertIsNotNone(actor)

        # Inspect PhysX component
        comp = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        self.assertIsNotNone(comp)
        self.assertEqual(len(comp.collision_shapes), 1)

        shape = comp.collision_shapes[0]
        self.assertEqual(shape.vertices.shape, (6, 3))
        self.assertEqual(shape.triangles.shape, (8, 3))

        # PhysX center of mass should be centered at (0, 0, 0)
        cm_pos = comp.cmass_local_pose.p
        self.assertLess(np.linalg.norm(cm_pos), 1e-5)

    def test_cli_execution_subprocess(self):
        """Verify CLI execution via subprocess runs cleanly with exit code 0."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_out = Path(tmp_dir) / "cli_test_prism.obj"
            cmd = [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "generate_prism_mesh.py"),
                "--output", str(test_out),
                "--base-edge", "0.035",
                "--height", "0.035",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"CLI exited with error: {result.stderr}")
            self.assertTrue(test_out.is_file())
            self.assertIn("SUCCESS: Triangular prism mesh generated and validated", result.stdout)


if __name__ == "__main__":
    unittest.main()
