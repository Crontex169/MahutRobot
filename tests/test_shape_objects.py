"""
Unit test suite for Milestone 2: Modular 9 Shape-Color Object Factory.

Verifies:
1. Factory registry keys and naming ({color}_{shape})
2. Visual RenderMaterial RGBA colors and IEEE 754 tolerance
3. PhysX contact materials (friction=2.0, restitution=0.0)
4. Collision shape geometries and dimensions (35 mm scale)
5. Physical mass (0.035 kg), inertia tensor, and center of mass alignment
6. Underground pre-parking initial pose (z = -10.0 m)
7. Collision group isolation against ground plane depenetration
8. Dynamic mesh path resolution and exception handling
9. Individual shape builder functions (cube, cylinder, triangular prism)
10. ManiSkill scene and Actor wrapper integration
11. PhysX dynamic simulation stability and settling behavior
"""

import math
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np
import sapien
import sapien.physx as physx
import sapien.render as render

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mahut_kol.envs.shape_objects as so
from mahut_kol.envs.shape_objects import (
    CUBE_HALF_SIZE,
    CYLINDER_HALF_LENGTH,
    CYLINDER_RADIUS,
    DEFAULT_MASS,
    DEFAULT_PARKING_Z,
    OBJECT_COLORS,
    OBJECT_SHAPES,
    PRISM_BASE_EDGE,
    PRISM_HEIGHT,
    SHAPE_DENSITIES,
    build_cube,
    build_cylinder,
    build_manipulation_objects,
    build_triangular_prism,
    get_entity,
    get_prism_mesh_path,
)


class TestShapeObjectsFactory(unittest.TestCase):
    """Unit tests for shape_objects factory module."""

    def setUp(self):
        self.scene = sapien.Scene()
        self.objects = build_manipulation_objects(self.scene)

    def test_factory_keys_and_naming(self):
        """Verify all 9 keys exist in {color}_{shape} format and actor names match."""
        expected_keys = {
            f"{c}_{s}" for c in ("red", "green", "blue") for s in ("cube", "cylinder", "triangular_prism")
        }
        self.assertEqual(len(self.objects), 9)
        self.assertEqual(set(self.objects.keys()), expected_keys)

        for key, actor in self.objects.items():
            entity = get_entity(actor)
            self.assertEqual(entity.name, key)

    def test_render_material_colors(self):
        """Verify RenderMaterial RGBA colors match exact specifications within float32 tolerance."""
        for key, actor in self.objects.items():
            color_name, shape_name = key.split("_", 1)
            expected_rgba = OBJECT_COLORS[color_name]

            entity = get_entity(actor)
            render_body = entity.find_component_by_type(render.RenderBodyComponent)
            self.assertIsNotNone(render_body, f"RenderBodyComponent missing on {key}")
            self.assertGreater(len(render_body.render_shapes), 0)

            actual_rgba = render_body.render_shapes[0].material.base_color
            np.testing.assert_allclose(
                actual_rgba,
                expected_rgba,
                atol=1e-6,
                err_msg=f"RGBA color mismatch for {key}: expected {expected_rgba}, got {actual_rgba}",
            )
            # Alpha channel must be strictly 1.0
            self.assertAlmostEqual(actual_rgba[3], 1.0, places=6)

    def test_physx_materials(self):
        """Verify PhysxMaterial properties: static_friction=2.0, dynamic_friction=2.0, restitution=0.0."""
        for key, actor in self.objects.items():
            entity = get_entity(actor)
            dynamic_comp = entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
            self.assertIsNotNone(dynamic_comp, f"PhysxRigidDynamicComponent missing on {key}")
            self.assertGreater(len(dynamic_comp.collision_shapes), 0)

            shape = dynamic_comp.collision_shapes[0]
            mat = shape.physical_material
            self.assertAlmostEqual(mat.static_friction, 2.0, places=5)
            self.assertAlmostEqual(mat.dynamic_friction, 2.0, places=5)
            self.assertAlmostEqual(mat.restitution, 0.0, places=5)

    def test_geometric_dimensions(self):
        """Verify collision shape geometries and bounding dimensions conform to 35 mm scale."""
        for key, actor in self.objects.items():
            entity = get_entity(actor)
            dynamic_comp = entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
            shape = dynamic_comp.collision_shapes[0]

            if "cube" in key:
                self.assertIsInstance(shape, physx.PhysxCollisionShapeBox)
                np.testing.assert_allclose(shape.half_size, CUBE_HALF_SIZE, atol=1e-6)
            elif "cylinder" in key:
                self.assertIsInstance(shape, physx.PhysxCollisionShapeCylinder)
                self.assertAlmostEqual(shape.radius, CYLINDER_RADIUS, places=6)
                self.assertAlmostEqual(shape.half_length, CYLINDER_HALF_LENGTH, places=6)
            elif "triangular_prism" in key:
                self.assertIsInstance(shape, physx.PhysxCollisionShapeConvexMesh)
                verts = shape.vertices
                faces = shape.triangles
                self.assertEqual(verts.shape, (6, 3))
                self.assertEqual(faces.shape, (8, 3))

                # Check bounding extents
                x_span = verts[:, 0].max() - verts[:, 0].min()
                z_span = verts[:, 2].max() - verts[:, 2].min()
                self.assertAlmostEqual(x_span, PRISM_BASE_EDGE, places=5)
                self.assertAlmostEqual(z_span, PRISM_HEIGHT, places=5)

    def test_object_mass_and_inertia(self):
        """Verify rigid dynamic mass equals 0.035 +/- 0.001 kg, COM at origin, and positive inertia."""
        for key, actor in self.objects.items():
            entity = get_entity(actor)
            dynamic_comp = entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
            mass = dynamic_comp.mass
            self.assertAlmostEqual(mass, DEFAULT_MASS, delta=0.001, msg=f"Mass mismatch on {key}")

            # Center of mass offset relative to actor origin
            com_pose = dynamic_comp.cmass_local_pose
            com_dist = np.linalg.norm(com_pose.p)
            self.assertLess(com_dist, 1e-4, f"COM offset too large on {key}: {com_dist}")

            # Principal moments of inertia must be strictly positive
            inertia = dynamic_comp.inertia
            self.assertTrue(np.all(inertia > 0), f"Inertia not positive definite on {key}: {inertia}")

    def test_initial_parking_pose(self):
        """Verify actors are pre-parked at z = -10.0 m with identity orientation."""
        for key, actor in self.objects.items():
            entity = get_entity(actor)
            p = entity.pose.p
            q = entity.pose.q

            self.assertAlmostEqual(p[0], 0.0, places=4)
            self.assertAlmostEqual(p[1], 0.0, places=4)
            self.assertAlmostEqual(p[2], DEFAULT_PARKING_Z, places=4)

            # Identity quaternion [1, 0, 0, 0]
            np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-5)

    def test_collision_group_isolation(self):
        """Verify pre_disable_collisions sets collision groups to 0, and can be re-enabled."""
        # Objects built with pre_disable_collisions=True
        for key, actor in self.objects.items():
            entity = get_entity(actor)
            dynamic_comp = entity.find_component_by_type(physx.PhysxRigidDynamicComponent)
            shape = dynamic_comp.collision_shapes[0]
            cg = shape.get_collision_groups()
            self.assertEqual(cg[0], 0, f"Collision group 0 not isolated for {key}")
            self.assertEqual(cg[1], 0, f"Collision group 1 not isolated for {key}")

        # Objects built with pre_disable_collisions=False have default groups [1, 1, 0, 0]
        test_scene = sapien.Scene()
        active_cube = build_cube(
            test_scene,
            color=[0.85, 0.15, 0.15, 1.0],
            name="active_cube",
            pre_disable_collisions=False,
        )
        entity_active = get_entity(active_cube)
        shape_active = entity_active.find_component_by_type(physx.PhysxRigidDynamicComponent).collision_shapes[0]
        cg_active = shape_active.get_collision_groups()
        self.assertEqual(cg_active[0], 1)
        self.assertEqual(cg_active[1], 1)

    def test_dynamic_mesh_path_resolution(self):
        """Verify get_prism_mesh_path locates asset, supports custom paths, and raises on missing."""
        default_path = get_prism_mesh_path()
        self.assertTrue(default_path.is_file())
        self.assertEqual(default_path.name, "triangular_prism.obj")

        # Explicit existing path
        explicit_path = get_prism_mesh_path(str(default_path))
        self.assertEqual(explicit_path, default_path)

        # Invalid path raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            get_prism_mesh_path("/non/existent/path/triangular_prism.obj")

    def test_individual_shape_builders(self):
        """Verify individual helper builders (build_cube, build_cylinder, build_triangular_prism)."""
        scene = sapien.Scene()

        cube = build_cube(scene, [0.85, 0.15, 0.15, 1.0], "my_cube", initial_parking_z=-5.0)
        cyl = build_cylinder(scene, [0.15, 0.80, 0.20, 1.0], "my_cyl", initial_parking_z=-5.0)
        prism = build_triangular_prism(scene, [0.15, 0.30, 0.85, 1.0], "my_prism", initial_parking_z=-5.0)

        for obj, shape_type in [
            (cube, physx.PhysxCollisionShapeBox),
            (cyl, physx.PhysxCollisionShapeCylinder),
            (prism, physx.PhysxCollisionShapeConvexMesh),
        ]:
            ent = get_entity(obj)
            comp = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
            self.assertIsInstance(comp.collision_shapes[0], shape_type)
            self.assertAlmostEqual(comp.mass, DEFAULT_MASS, delta=0.001)
            self.assertAlmostEqual(ent.pose.p[2], -5.0, places=4)

    def test_maniskill_scene_integration(self):
        """Verify integration inside ManiSkillScene and Actor wrapper."""
        import gymnasium as gym
        import mahut_kol.envs
        from mani_skill.utils.structs.actor import Actor

        env = gym.make("MahutKol-Minimal-v0")
        ms_scene = env.unwrapped.scene
        actors = build_manipulation_objects(ms_scene)

        self.assertEqual(len(actors), 9)
        for key, act in actors.items():
            self.assertIsInstance(act, Actor, f"{key} should be an Actor instance")
            self.assertEqual(act.name, key)
            self.assertAlmostEqual(act.mass.item(), DEFAULT_MASS, delta=0.001)
            # Pose in ManiSkill Actor is batched tensor
            pos = act.pose.p.squeeze().tolist()
            self.assertAlmostEqual(pos[2], DEFAULT_PARKING_Z, places=3)

        env.close()

    def test_dynamic_simulation_drop_stability(self):
        """Verify dynamic settling onto ground plane: no NaNs, stable velocities and heights."""
        scene = sapien.Scene()
        scene.set_timestep(1.0 / 60.0)

        # Ground plane facing +Z
        gb = scene.create_actor_builder()
        mat = physx.PhysxMaterial(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        gb.add_plane_collision(
            sapien.Pose(p=[0, 0, 0], q=[0.7071068, 0, -0.7071068, 0]),
            material=mat,
        )
        gb.build_static("ground")

        # Spawn active cube, cylinder, and prism dropping from z = 0.10 m
        cube = build_cube(
            scene,
            [0.85, 0.15, 0.15, 1.0],
            "drop_cube",
            initial_parking_z=0.10,
            pre_disable_collisions=False,
        )
        cyl = build_cylinder(
            scene,
            [0.15, 0.80, 0.20, 1.0],
            "drop_cyl",
            initial_parking_z=0.10,
            pre_disable_collisions=False,
        )
        prism = build_triangular_prism(
            scene,
            [0.15, 0.30, 0.85, 1.0],
            "drop_prism",
            initial_parking_z=0.10,
            pre_disable_collisions=False,
        )

        # Separate positions along X axis
        get_entity(cube).set_pose(sapien.Pose(p=[-0.10, 0.0, 0.10]))
        get_entity(cyl).set_pose(sapien.Pose(p=[0.0, 0.0, 0.10]))
        get_entity(prism).set_pose(sapien.Pose(p=[0.10, 0.0, 0.10]))

        # Simulate 150 steps (~2.5 seconds)
        for _ in range(150):
            scene.step()

        for name, obj in [("cube", cube), ("cylinder", cyl), ("prism", prism)]:
            ent = get_entity(obj)
            comp = ent.find_component_by_type(physx.PhysxRigidDynamicComponent)
            pos = ent.pose.p
            lin_vel = comp.linear_velocity

            self.assertFalse(np.isnan(pos).any(), f"NaN detected in position of {name}")
            self.assertFalse(np.isinf(pos).any(), f"Inf detected in position of {name}")

            # Linear velocity should be near zero (settled on ground)
            speed = np.linalg.norm(lin_vel)
            self.assertLess(speed, 0.05, f"{name} did not settle stably: speed = {speed}")

            # Resting height should be above ground and around half-size
            self.assertGreater(pos[2], 0.010, f"{name} penetrated ground plane: z = {pos[2]}")
            self.assertLess(pos[2], 0.030, f"{name} bounced or flew upward: z = {pos[2]}")


if __name__ == "__main__":
    unittest.main()
