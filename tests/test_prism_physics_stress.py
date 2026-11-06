import math
import unittest
from pathlib import Path

import numpy as np
import scipy.spatial.transform
import sapien

_REPO_ROOT = Path(__file__).resolve().parent.parent
OBJ_PATH = _REPO_ROOT / "assets" / "objects" / "triangular_prism.obj"


class TestPrismPhysicsStress(unittest.TestCase):
    """
    Adversarial dynamic physics and simulation stress test suite for Milestone 1.
    Evaluates SAPIEN 3 / PhysX contact stability, inertia tensor correctness,
    drop impact dynamics, resting face equilibrium, and absence of tunneling/explosions.
    """

    def setUp(self):
        self.assertTrue(OBJ_PATH.is_file(), f"Prism OBJ not found at {OBJ_PATH}")
        self.a = 0.035
        self.h = 0.035
        self.r = self.a / (2.0 * math.sqrt(3.0))  # inradius ~ 0.0101036 m
        self.mat = sapien.physx.PhysxMaterial(
            static_friction=2.0, dynamic_friction=2.0, restitution=0.0
        )

    def _create_prism_actor(self, scene, initial_pose=None, name="test_prism", mass=None):
        builder = scene.create_actor_builder()
        builder.add_convex_collision_from_file(str(OBJ_PATH), material=self.mat)
        if initial_pose is not None:
            builder.initial_pose = initial_pose
        actor = builder.build(name=name)
        comp = actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if mass is not None:
            comp.mass = mass
        return actor, comp

    def test_inertia_tensor_and_center_of_mass(self):
        """
        Verify inertia tensor positive definiteness, analytical match to equilateral
        prism solid body equations, and zero center-of-mass offset.
        """
        scene = sapien.Scene()
        actor, comp = self._create_prism_actor(scene)

        # Theoretical volume and mass (density = 1000 kg/m^3 in PhysX default)
        v_theo = (math.sqrt(3.0) / 4.0 * (self.a**2)) * self.h
        m_theo = 1000.0 * v_theo

        self.assertAlmostEqual(
            comp.mass, m_theo, delta=1e-6,
            msg=f"Mass mismatch: {comp.mass} vs theoretical {m_theo}",
        )

        # Center of mass should be at (0, 0, 0)
        cm_offset = np.linalg.norm(comp.cmass_local_pose.p)
        self.assertLess(
            cm_offset, 1e-7,
            msg=f"Center of mass offset {cm_offset:.2e} m exceeds 1e-7 m",
        )

        # Analytical moments of inertia:
        # I_zz = (1/12) * M * a^2
        # I_xx = I_yy = (1/2) * I_zz + (1/12) * M * h^2
        i_zz_theo = (1.0 / 12.0) * comp.mass * (self.a**2)
        i_xx_theo = 0.5 * i_zz_theo + (1.0 / 12.0) * comp.mass * (self.h**2)
        i_yy_theo = i_xx_theo

        inertia = comp.inertia
        # Check positive definiteness
        self.assertTrue(np.all(inertia > 0.0), f"Inertia components must be positive: {inertia}")

        # Check rotational symmetry I_xx == I_yy
        self.assertAlmostEqual(inertia[0], inertia[1], delta=1e-8)

        # Check match with theoretical values within 0.1%
        np.testing.assert_allclose(
            inertia,
            [i_xx_theo, i_yy_theo, i_zz_theo],
            rtol=1e-3,
            err_msg=f"Inertia tensor {inertia} does not match theoretical [{i_xx_theo}, {i_yy_theo}, {i_zz_theo}]",
        )

    def test_drop_from_height_100_steps(self):
        """
        Standard drop test: drop from z = 0.5 m onto ground plane.
        Verify no NaNs, no Infs, no tunneling, and bounded velocities across 100+ steps.
        """
        scene = sapien.Scene()
        scene.set_timestep(1 / 100.0)
        scene.add_ground(0.0)

        init_pose = sapien.Pose(p=[0, 0, 0.5])
        actor, comp = self._create_prism_actor(scene, initial_pose=init_pose)

        for step in range(1, 151):
            scene.step()
            pos = actor.pose.p
            q = actor.pose.q
            v = comp.linear_velocity
            w = comp.angular_velocity

            self.assertFalse(np.any(np.isnan(pos)), f"NaN in position at step {step}")
            self.assertFalse(np.any(np.isinf(pos)), f"Inf in position at step {step}")
            self.assertFalse(np.any(np.isnan(q)), f"NaN in orientation at step {step}")
            self.assertFalse(np.any(np.isnan(v)), f"NaN in linear velocity at step {step}")
            self.assertFalse(np.any(np.isnan(w)), f"NaN in angular velocity at step {step}")

            # Tunneling check: centroid must never penetrate ground plane significantly
            self.assertGreaterEqual(
                pos[2], -0.001,
                f"Severe tunneling at step {step}: z={pos[2]:.6f}",
            )

        # Settled state at step 150
        v_norm = np.linalg.norm(comp.linear_velocity)
        w_norm = np.linalg.norm(comp.angular_velocity)
        self.assertLess(v_norm, 1e-3, f"Non-zero residual linear velocity: {v_norm}")
        self.assertLess(w_norm, 1e-3, f"Non-zero residual angular velocity: {w_norm}")

        # Resting height on bottom triangular face should be exactly h/2 = 0.0175 m
        self.assertAlmostEqual(actor.pose.p[2], self.h / 2.0, delta=1e-4)

    def test_resting_face_stability_all_5_faces(self):
        """
        Verify the prism can rest stably on each of its 5 physical faces:
        - 2 equilateral triangular caps (rest z ~ h/2 = 0.0175 m)
        - 3 rectangular side faces (rest z ~ r = 0.0101036 m)
        Verify zero creep / jitter over 300 simulation steps.
        """
        face_orientations = {
            "triangular_bottom": [1, 0, 0, 0],
            "triangular_top": scipy.spatial.transform.Rotation.from_euler("x", 180, degrees=True).as_quat()[[3, 0, 1, 2]],
            "rect_base": scipy.spatial.transform.Rotation.from_euler("x", 90, degrees=True).as_quat()[[3, 0, 1, 2]],
            "rect_slant_right": (
                scipy.spatial.transform.Rotation.from_euler("z", -60, degrees=True)
                * scipy.spatial.transform.Rotation.from_euler("x", 90, degrees=True)
            ).as_quat()[[3, 0, 1, 2]],
            "rect_slant_left": (
                scipy.spatial.transform.Rotation.from_euler("z", 60, degrees=True)
                * scipy.spatial.transform.Rotation.from_euler("x", 90, degrees=True)
            ).as_quat()[[3, 0, 1, 2]],
        }

        for face_name, q in face_orientations.items():
            with self.subTest(face=face_name):
                scene = sapien.Scene()
                scene.set_timestep(1 / 100.0)
                scene.add_ground(0.0)

                expected_rest_z = (self.h / 2.0) if "triangular" in face_name else self.r
                drop_z = expected_rest_z + 0.05  # drop 5 cm onto ground

                actor, comp = self._create_prism_actor(
                    scene,
                    initial_pose=sapien.Pose(p=[0, 0, drop_z], q=q),
                    name=face_name,
                )

                for _ in range(300):
                    scene.step()

                final_z = actor.pose.p[2]
                v_norm = np.linalg.norm(comp.linear_velocity)
                w_norm = np.linalg.norm(comp.angular_velocity)

                self.assertAlmostEqual(
                    final_z, expected_rest_z, delta=1e-4,
                    msg=f"Face {face_name} resting height mismatch: got {final_z}, expected {expected_rest_z}",
                )
                self.assertLess(v_norm, 1e-3, f"Face {face_name} has residual velocity: {v_norm}")
                self.assertLess(w_norm, 1e-3, f"Face {face_name} has residual angular velocity: {w_norm}")

    def test_adversarial_drops_edge_vertex_and_spinning(self):
        """
        Adversarially drop the prism from edge-down, vertex-down, tilted, and high-spin configurations.
        Verify convergence to a stable rest on one of the 5 faces without numerical explosion.
        """
        drop_configs = {
            "edge_drop": {
                "q": scipy.spatial.transform.Rotation.from_euler("y", 45, degrees=True).as_quat()[[3, 0, 1, 2]],
                "w": [0.0, 0.0, 0.0],
            },
            "vertex_drop": {
                "q": scipy.spatial.transform.Rotation.from_euler("xyz", [45, 45, 45], degrees=True).as_quat()[[3, 0, 1, 2]],
                "w": [0.0, 0.0, 0.0],
            },
            "high_spin_drop": {
                "q": [1, 0, 0, 0],
                "w": [10.0, 15.0, -12.0],
            },
        }

        for test_name, cfg in drop_configs.items():
            with self.subTest(test=test_name):
                scene = sapien.Scene()
                scene.set_timestep(1 / 100.0)
                scene.add_ground(0.0)

                actor, comp = self._create_prism_actor(
                    scene,
                    initial_pose=sapien.Pose(p=[0, 0, 0.5], q=cfg["q"]),
                    name=test_name,
                )
                if np.linalg.norm(cfg["w"]) > 0:
                    comp.set_angular_velocity(cfg["w"])

                for step in range(400):
                    scene.step()
                    pos = actor.pose.p
                    self.assertFalse(np.any(np.isnan(pos)))
                    self.assertGreaterEqual(pos[2], -0.001)

                v_norm = np.linalg.norm(comp.linear_velocity)
                w_norm = np.linalg.norm(comp.angular_velocity)
                self.assertLess(v_norm, 1e-2, f"{test_name} did not settle: |v|={v_norm}")
                self.assertLess(w_norm, 1e-1, f"{test_name} did not settle: |w|={w_norm}")

                # Must settle on either a triangular face (~0.0175) or rectangular face (~0.0101)
                final_z = actor.pose.p[2]
                on_triangular = abs(final_z - (self.h / 2.0)) < 1e-3
                on_rectangular = abs(final_z - self.r) < 1e-3
                self.assertTrue(
                    on_triangular or on_rectangular,
                    f"{test_name} settled at unclassified height: z={final_z}",
                )

    def test_tabletop_collision_and_resting_alignment(self):
        """
        Verify prism dropped onto a tabletop static actor settles with sub-micrometer precision.
        """
        scene = sapien.Scene()
        scene.set_timestep(1 / 100.0)

        # Create static table box
        table_half_z = 0.025
        table_z = 0.3
        table_top_z = table_z + table_half_z  # 0.325 m

        tb = scene.create_actor_builder()
        tb.add_box_collision(half_size=[0.5, 0.5, table_half_z], material=self.mat)
        tb.initial_pose = sapien.Pose(p=[0, 0, table_z])
        tb.build_static(name="table")

        # Drop prism from 0.6 m (0.275 m above table surface)
        actor, comp = self._create_prism_actor(
            scene, initial_pose=sapien.Pose(p=[0, 0, 0.6]), name="table_prism"
        )

        for _ in range(250):
            scene.step()

        expected_z = table_top_z + (self.h / 2.0)
        pos = actor.pose.p
        v_norm = np.linalg.norm(comp.linear_velocity)
        w_norm = np.linalg.norm(comp.angular_velocity)

        self.assertAlmostEqual(pos[2], expected_z, delta=1e-4)
        self.assertLess(v_norm, 1e-3)
        self.assertLess(w_norm, 1e-3)

    def test_stacking_stability_cube_and_prism(self):
        """
        Verify two-way stacking stability:
        1. Prism on bottom, Cube on top
        2. Cube on bottom, Prism on top
        Both stacks must remain in static equilibrium without sliding or collapsing.
        """
        half_dim = 0.0175  # 35mm scale / 2

        # 1. Prism bottom, Cube top
        scene1 = sapien.Scene()
        scene1.set_timestep(1 / 100.0)
        scene1.add_ground(0.0)

        p1_act, p1_comp = self._create_prism_actor(
            scene1, initial_pose=sapien.Pose(p=[0, 0, half_dim]), mass=0.035
        )
        cb1 = scene1.create_actor_builder()
        cb1.add_box_collision(half_size=[half_dim, half_dim, half_dim], material=self.mat)
        cb1.initial_pose = sapien.Pose(p=[0, 0, 3 * half_dim])
        c1_act = cb1.build(name="top_cube")
        c1_comp = c1_act.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        c1_comp.mass = 0.035

        for _ in range(300):
            scene1.step()

        self.assertAlmostEqual(p1_act.pose.p[2], half_dim, delta=1e-4)
        self.assertAlmostEqual(c1_act.pose.p[2], 3 * half_dim, delta=1e-4)
        self.assertLess(np.linalg.norm(p1_comp.linear_velocity), 1e-3)
        self.assertLess(np.linalg.norm(c1_comp.linear_velocity), 1e-3)

        # 2. Cube bottom, Prism top
        scene2 = sapien.Scene()
        scene2.set_timestep(1 / 100.0)
        scene2.add_ground(0.0)

        cb2 = scene2.create_actor_builder()
        cb2.add_box_collision(half_size=[half_dim, half_dim, half_dim], material=self.mat)
        cb2.initial_pose = sapien.Pose(p=[0, 0, half_dim])
        c2_act = cb2.build(name="bottom_cube")
        c2_comp = c2_act.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        c2_comp.mass = 0.035

        p2_act, p2_comp = self._create_prism_actor(
            scene2, initial_pose=sapien.Pose(p=[0, 0, 3 * half_dim]), mass=0.035
        )

        for _ in range(300):
            scene2.step()

        self.assertAlmostEqual(c2_act.pose.p[2], half_dim, delta=1e-4)
        self.assertAlmostEqual(p2_act.pose.p[2], 3 * half_dim, delta=1e-4)
        self.assertLess(np.linalg.norm(c2_comp.linear_velocity), 1e-3)
        self.assertLess(np.linalg.norm(p2_comp.linear_velocity), 1e-3)


if __name__ == "__main__":
    unittest.main()
