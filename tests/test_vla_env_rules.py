"""
Aşama 1 (VLA veri toplama) ortam kuralları:
- URDF kinematiği SAPIEN TCP pozuyla birebir aynı
- Çalışma alanı içinde IK ile kavrama erişilebilirliği
- TOP + üçgen prizma referansı geçersiz, 336 geçerli kombinasyon
- Hedef noktası boş ve çalışma alanı içinde
- Başarı = hedefte + bırakılmış + durağan (anlık mesafe yetmez)
- 224x224 front + wrist kameraları, 600 adım
"""

import itertools
import unittest

import gymnasium as gym
import numpy as np
import torch

import mahut_kol.envs  # noqa: F401
import mahut_kol.robots  # noqa: F401
from mahut_kol.envs.multi_shape_task import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_RELATIONAL_DELTA,
    MIN_PAIRWISE_DISTANCE,
    OBJECT_IDS,
    RELATIONS,
    SETTLE_STEPS,
    SUCCESS_XY_TOLERANCE,
    TABLETOP_HALF_HEIGHT,
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
    compute_relational_target_position,
    is_valid_command,
    valid_command_combinations,
    validate_command,
)
from mahut_kol.robots.kinematics import GRASP_TILTS_DEG, get_kinematics, solve_grasp_ik


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class TestCommandRules(unittest.TestCase):
    def test_336_valid_combinations(self):
        combos = valid_command_combinations()
        self.assertEqual(len(combos), 9 * 8 * 5 - 24)
        self.assertEqual(len(combos), 336)
        self.assertEqual(len(set(combos)), 336)
        for m, r, rel in combos:
            self.assertNotEqual(m, r)
            self.assertFalse(rel == "TOP" and r.endswith("triangular_prism"))

    def test_top_on_prism_raises_turkish(self):
        with self.assertRaises(ValueError) as ctx:
            validate_command("red_cube", "green_triangular_prism", "TOP")
        self.assertIn("prizma", str(ctx.exception))
        # prizma manipüle nesne olarak TOP'a konabilir
        self.assertTrue(is_valid_command("green_triangular_prism", "red_cube", "TOP"))

    def test_same_object_raises(self):
        with self.assertRaises(ValueError):
            validate_command("red_cube", "red_cube", "LEFT")

    def test_target_offsets_delta(self):
        self.assertAlmostEqual(DEFAULT_RELATIONAL_DELTA, 0.06)
        ref = np.array([0.15, 0.0, TABLETOP_HALF_HEIGHT])
        self.assertAlmostEqual(compute_relational_target_position(ref, "RIGHT")[1], 0.06, places=5)
        self.assertAlmostEqual(compute_relational_target_position(ref, "FRONT")[0], 0.09, places=5)

    def test_success_regions_do_not_overlap(self):
        # Komşu yön hedefleri arası mesafe > 2 * tolerans
        ref = np.zeros(3)
        planar = [r for r in RELATIONS if r != "TOP"]
        for a, b in itertools.combinations(planar, 2):
            d = np.linalg.norm(
                compute_relational_target_position(ref, a)[:2] - compute_relational_target_position(ref, b)[:2]
            )
            self.assertGreater(d, 2 * SUCCESS_XY_TOLERANCE)
        # Toleransın kenarındaki nesne referansa geçmez (35 mm nesneler)
        self.assertGreaterEqual(DEFAULT_RELATIONAL_DELTA - SUCCESS_XY_TOLERANCE, 0.035 - 1e-9)


class TestKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make("MahutKol-MultiShapeRelational-v1", obs_mode="state", control_mode="pd_joint_pos")
        cls.env.reset(seed=0)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_fk_matches_sapien(self):
        K = get_kinematics()
        u = self.env.unwrapped
        rng = np.random.default_rng(1)
        for _ in range(30):
            q = rng.uniform(K.q_min, K.q_max)
            u.agent.robot.set_qpos(torch.tensor([list(q) + [0.0, 0.0]], dtype=torch.float32))
            p_sim = u.agent.tcp.pose.p[0].numpy()
            R_sim = _quat_to_mat(u.agent.tcp.pose.q[0].numpy())
            p, R = K.fk(q)
            np.testing.assert_allclose(p, p_sim, atol=1e-5)
            np.testing.assert_allclose(R, R_sim, atol=1e-5)

    def test_fk_batch_matches_fk(self):
        K = get_kinematics()
        Q = np.random.default_rng(2).uniform(K.q_min, K.q_max, size=(20, 5))
        P, A = K.fk_batch(Q)
        for i, q in enumerate(Q):
            p, R = K.fk(q)
            np.testing.assert_allclose(P[i], p, atol=1e-9)
            np.testing.assert_allclose(A[i], K.approach_of(R), atol=1e-9)

    def test_arm_heading_matches_ik(self):
        from mahut_kol.envs.multi_shape_task import arm_heading

        K = get_kinematics()
        for x, y in [(0.12, -0.1), (0.2, 0.1), (0.16, -0.02)]:
            q, ok = solve_grasp_ik([x, y, 0.022], np.radians(50))
            self.assertTrue(ok)
            self.assertAlmostEqual(arm_heading(x, y), q[0], delta=np.radians(1.0))

    def test_workspace_corners_reachable(self):
        """Çalışma alanı köşeleri ve merkezi en az bir kavrama eğimiyle erişilebilir."""
        pts = [
            (x, y)
            for x in (WORKSPACE_X_MIN, WORKSPACE_X_MAX, (WORKSPACE_X_MIN + WORKSPACE_X_MAX) / 2)
            for y in (WORKSPACE_Y_MIN, WORKSPACE_Y_MAX, (WORKSPACE_Y_MIN + WORKSPACE_Y_MAX) / 2)
        ]
        for x, y in pts:
            ok_any = False
            for t in GRASP_TILTS_DEG:
                _, ok = solve_grasp_ik([x, y, 0.022], np.radians(t))
                if ok:
                    ok_any = True
                    break
            self.assertTrue(ok_any, f"({x:.2f}, {y:.2f}) erişilemiyor")


class TestEnvRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gym.make(
            "MahutKol-MultiShapeRelational-v1", obs_mode="rgb", control_mode="pd_joint_pos"
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_max_episode_steps(self):
        self.assertEqual(DEFAULT_MAX_EPISODE_STEPS, 600)
        self.assertEqual(self.env._max_episode_steps, 600)

    def test_cameras(self):
        obs, _ = self.env.reset(seed=1)
        for cam in ("front", "wrist"):
            self.assertEqual(
                tuple(obs["sensor_data"][cam]["rgb"].shape[1:]), (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
            )
        self.assertEqual((CAMERA_WIDTH, CAMERA_HEIGHT), (224, 224))

    def test_reset_rejects_top_on_prism(self):
        with self.assertRaises(ValueError):
            self.env.reset(
                options=dict(manipulated_object="red_cube", reference_object="blue_triangular_prism", relation="TOP")
            )

    def test_random_reset_never_samples_invalid(self):
        u = self.env.unwrapped
        for seed in range(40):
            self.env.reset(seed=seed)
            self.assertTrue(is_valid_command(u.manipulated_object_id, u.reference_object_id, u.current_relation))
            self.assertEqual(len(u.distractor_object_ids), 2)

    def test_target_free_and_in_workspace(self):
        u = self.env.unwrapped
        for seed in range(30):
            rel = RELATIONS[seed % 5]
            ref = "red_cylinder" if rel == "TOP" else OBJECT_IDS[(seed * 7) % 9]
            man = "blue_cube" if ref != "blue_cube" else "green_cube"
            self.env.reset(
                seed=seed, options=dict(manipulated_object=man, reference_object=ref, relation=rel)
            )
            tgt = u.get_relational_target_position()[0].numpy()
            self.assertGreaterEqual(tgt[0], WORKSPACE_X_MIN - 1e-6)
            self.assertLessEqual(tgt[0], WORKSPACE_X_MAX + 1e-6)
            self.assertGreaterEqual(tgt[1], WORKSPACE_Y_MIN - 1e-6)
            self.assertLessEqual(tgt[1], WORKSPACE_Y_MAX + 1e-6)
            others = [u.manipulated_object_id] + u.distractor_object_ids
            for k in others:
                p = u.manipulation_objects[k].pose.p[0].numpy()
                if rel != "TOP":
                    self.assertGreaterEqual(np.linalg.norm(p[:2] - tgt[:2]), MIN_PAIRWISE_DISTANCE - 1e-6)
            act = [u.manipulated_object_id, u.reference_object_id] + u.distractor_object_ids
            pts = [u.manipulation_objects[k].pose.p[0].numpy()[:2] for k in act]
            for a, b in itertools.combinations(pts, 2):
                self.assertGreaterEqual(np.linalg.norm(a - b), MIN_PAIRWISE_DISTANCE - 1e-6)

    def _teleport_to_target(self, finger_q):
        """Nesneyi hedefe koy, kıskacı yukarıda tut, parmakları finger_q'ya ayarla."""
        u = self.env.unwrapped
        tgt = u.get_relational_target_position()
        obj = u.get_manipulated_object()
        obj.set_pose(type(obj.pose).create_from_pq(p=tgt, q=torch.tensor([[1.0, 0, 0, 0]])))
        qpos = u.agent.robot.get_qpos().clone()
        qpos[:, 5:7] = finger_q
        u.agent.robot.set_qpos(qpos)
        return qpos

    def test_success_requires_release_and_settling(self):
        u = self.env.unwrapped
        self.env.reset(
            seed=5, options=dict(manipulated_object="blue_cube", reference_object="red_cube", relation="LEFT")
        )
        info = u.evaluate()
        self.assertFalse(bool(info["success"]))
        qpos = self._teleport_to_target(0.0)
        # Anlık: konumda ama henüz durağanlık sayacı dolmadı
        info = u.evaluate()
        self.assertTrue(bool(info["is_placed"]))
        self.assertFalse(bool(info["success"]))
        succ = []
        for _ in range(SETTLE_STEPS + 2):
            _, _, _, _, info = self.env.step(qpos[0])
            succ.append(bool(info["success"]))
        self.assertFalse(succ[0])
        self.assertTrue(succ[-1], info)

    def test_no_success_when_gripper_closed(self):
        u = self.env.unwrapped
        self.env.reset(
            seed=6, options=dict(manipulated_object="blue_cube", reference_object="red_cube", relation="RIGHT")
        )
        qpos = self._teleport_to_target(0.042)  # kapalı kıskaç = "bırakılmadı"
        for _ in range(SETTLE_STEPS + 2):
            _, _, _, _, info = self.env.step(qpos[0])
        self.assertTrue(bool(info["is_placed"]))
        self.assertFalse(bool(info["is_released"]))
        self.assertFalse(bool(info["success"]))


if __name__ == "__main__":
    unittest.main()
