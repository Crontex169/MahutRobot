#!/usr/bin/env python3
"""
MuJoCo SO-100 Webcam Teleoperasyon Modülü Test Paketi
"""

import math
import os
import sys
import unittest
import numpy as np

# Side_project dizinini sys.path'e ekle
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ik_solver import SO100IKSolver
try:
    import mujoco
except ImportError:
    mujoco = None

# teleop_webcam modülünden test edilecek matematiksel ve mantıksal fonksiyonları içe aktar
try:
    import teleop_webcam
    from teleop_webcam import (
        CONFIG,
        compute_pinch_ratio,
        compute_gripper_value,
        compute_hand_orientation,
        apply_ema,
        apply_step_limiter,
        clamp_to_workspace,
        map_camera_deltas_to_robot,
        gripper_to_jaw_angle,
    )
except ImportError:
    teleop_webcam = None


class TestWebcamTeleopLogic(unittest.TestCase):
    """Webcam teleoperasyon haritalama, filtreleme ve gripper hesaplama testleri."""

    def setUp(self):
        if teleop_webcam is None:
            self.skipTest("teleop_webcam henüz oluşturulmadı (RED aşaması).")

    def test_gripper_value_bounds(self):
        """Pinch oranı sınır değerlerinde gripper'ın tam -1.0 ve 1.0 dönmesini test et."""
        # Pinch kapalı (<= pinch_min)
        val_closed = compute_gripper_value(0.10, pinch_min=0.20, pinch_max=0.75)
        self.assertAlmostEqual(val_closed, -1.0, places=4)

        # Pinch açık (>= pinch_max)
        val_open = compute_gripper_value(0.90, pinch_min=0.20, pinch_max=0.75)
        self.assertAlmostEqual(val_open, 1.0, places=4)

        # Orta nokta
        mid_pinch = 0.5 * (0.20 + 0.75)
        val_mid = compute_gripper_value(mid_pinch, pinch_min=0.20, pinch_max=0.75)
        self.assertAlmostEqual(val_mid, 0.0, places=2)
        self.assertTrue(-1.0 <= val_mid <= 1.0)

    def test_gripper_monotonic(self):
        """Pinch mesafesi arttıkça gripper açıklığının monoton artması gerekir."""
        pinches = np.linspace(0.1, 0.8, 15)
        vals = [compute_gripper_value(p, pinch_min=0.20, pinch_max=0.75) for p in pinches]
        for i in range(len(vals) - 1):
            self.assertLessEqual(vals[i], vals[i + 1])

    def test_gripper_to_jaw_angle(self):
        """Gripper [-1.0, 1.0] aralığından jaw açı aralığına doğrusal dönüşüm."""
        jaw_min, jaw_max = -0.174, 1.75
        self.assertAlmostEqual(gripper_to_jaw_angle(-1.0, jaw_min, jaw_max), jaw_min, places=4)
        self.assertAlmostEqual(gripper_to_jaw_angle(1.0, jaw_min, jaw_max), jaw_max, places=4)
        self.assertAlmostEqual(gripper_to_jaw_angle(0.0, jaw_min, jaw_max), 0.5 * (jaw_min + jaw_max), places=4)

    def test_ema_filter(self):
        """EMA filtresinin yumuşatma davranışı."""
        prev = np.array([0.0, 0.0, 0.0])
        new = np.array([1.0, 1.0, 1.0])
        alpha = 0.4
        filtered = apply_ema(new, prev, alpha)
        expected = alpha * new + (1.0 - alpha) * prev
        np.testing.assert_allclose(filtered, expected)

    def test_step_limiter(self):
        """Maksimum sıçrama sınırının büyük hareketleri kısıtlaması."""
        current = np.array([0.0, 0.0, 0.0])
        candidate_far = np.array([0.1, 0.0, 0.0])  # 10 cm sıçrama
        max_step = 0.035                           # 3.5 cm azami adım

        limited = apply_step_limiter(candidate_far, current, max_step)
        dist = np.linalg.norm(limited - current)
        self.assertAlmostEqual(dist, max_step, places=4)
        self.assertAlmostEqual(limited[0], 0.035, places=4)

        # Küçük adım kırpılmamalıdır
        candidate_near = np.array([0.01, 0.0, 0.0])
        kept = apply_step_limiter(candidate_near, current, max_step)
        np.testing.assert_allclose(kept, candidate_near)

    def test_workspace_clamping(self):
        """Çalışma hacmi dışına çıkan koordinatların sınırlandırılması."""
        ws = {
            "x": (-0.28, 0.28),
            "y": (-0.48, -0.05),
            "z": (0.015, 0.42),
        }
        out_of_bounds = np.array([0.50, 0.20, -0.10])
        clamped = clamp_to_workspace(out_of_bounds, ws)
        self.assertEqual(clamped[0], 0.28)
        self.assertEqual(clamped[1], -0.05)
        self.assertEqual(clamped[2], 0.015)

    def test_map_camera_deltas(self):
        """Kamera hareketlerinin robot dünyasındaki doğru eksenlere yönlendirilmesi."""
        # Sağa hareket (+X kamera): Robot +X
        cfg = CONFIG.copy()
        d_pos_right = map_camera_deltas_to_robot(
            dx_cam=0.1, dy_cam=0.0,
            hand_width=100.0, anchor_hand_width=100.0,
            config=cfg
        )
        self.assertGreater(d_pos_right[0], 0.0, "Sağa hareket robot X ekseninde pozitif olmalıdır.")
        self.assertAlmostEqual(d_pos_right[1], 0.0)
        self.assertAlmostEqual(d_pos_right[2], 0.0)

        # Yukarı hareket (-Y kamera piksellerinde yukarı): Robot +Z
        d_pos_up = map_camera_deltas_to_robot(
            dx_cam=0.0, dy_cam=-0.1,
            hand_width=100.0, anchor_hand_width=100.0,
            config=cfg
        )
        self.assertGreater(d_pos_up[2], 0.0, "Yukarı hareket robot Z ekseninde pozitif olmalıdır.")

        # Yaklaşma (el boyutu büyümesi): Robot ileri (-Y MuJoCo ekseni)
        d_pos_fwd = map_camera_deltas_to_robot(
            dx_cam=0.0, dy_cam=0.0,
            hand_width=120.0, anchor_hand_width=100.0,
            config=cfg
        )
        self.assertLess(d_pos_fwd[1], 0.0, "Elin yaklaşması robotu ileri (-Y) yönlendirmelidir.")

    def test_hand_orientation_math(self):
        """Sentetik el noktaları ile Pitch ve Roll açı hesaplamalarının doğrulanması."""
        # Düz duran el (avuç kameraya dönük)
        wrist = np.array([0.5, 0.8, 0.0])
        middle_mcp = np.array([0.5, 0.4, 0.0])
        index_mcp = np.array([0.4, 0.45, 0.0])
        pinky_mcp = np.array([0.6, 0.45, 0.0])

        pitch, roll = compute_hand_orientation(wrist, index_mcp, middle_mcp, pinky_mcp)
        self.assertTrue(math.isfinite(pitch))
        self.assertTrue(math.isfinite(roll))

    def test_ik_integration(self):
        """SO-100 DLS IK çözücüsünün teleop sahnesi ile doğru çalışması."""
        if mujoco is None:
            self.skipTest("mujoco kütüphanesi bulunamadı.")

        scene_path = os.path.join(CURRENT_DIR, "so_arm100", "teleop_scene.xml")
        self.assertTrue(os.path.exists(scene_path), f"Sahne dosyası eksik: {scene_path}")

        model = mujoco.MjModel.from_xml_path(scene_path)
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)

        ik = SO100IKSolver(model)
        ee_pos, _ = ik.get_ee_pose(data)

        # Home pozisyonuna yakın küçük bir hedef ver
        target = ee_pos + np.array([0.02, -0.02, 0.01])
        q_sol, conv = ik.solve(target_pos=target, q_init=data.qpos[:5], n_iters=10)
        self.assertTrue(len(q_sol) == 5)
        # Eklem limitleri içinde olmalı
        for i in range(5):
            self.assertGreaterEqual(q_sol[i], ik.q_min[i] - 1e-3)
            self.assertLessEqual(q_sol[i], ik.q_max[i] + 1e-3)

    def test_hud_rendering(self):
        """WebcamTeleopHUD sınıfının sentetik görüntü üzerine sorunsuz çizim yapması."""
        from teleop_webcam import WebcamTeleopHUD
        hud = WebcamTeleopHUD()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Clutch aktif ve pasif durumlarında render testleri
        out_active = hud.render(
            frame=dummy_frame.copy(),
            clutch_active=True,
            gripper_val=0.5,
            target_pos=np.array([0.05, -0.25, 0.12]),
            fps=32.4,
            hand_detected=True,
            ref_pt_px=(320, 240),
            thumb_px=(300, 200),
            index_px=(310, 190),
            clutch_mode="toggle",
            wrist_active=True,
        )
        self.assertEqual(out_active.shape, (480, 640, 3))

        out_passive = hud.render(
            frame=dummy_frame.copy(),
            clutch_active=False,
            gripper_val=-1.0,
            target_pos=np.array([0.0, -0.24, 0.08]),
            fps=28.0,
            hand_detected=False,
            clutch_mode="hold",
            wrist_active=False,
        )
        self.assertEqual(out_passive.shape, (480, 640, 3))

    def test_depth_deadband(self):
        """El boyutundaki küçük titreşimlerin ölü bölge (deadband) ile filtrelenmesi."""
        cfg = CONFIG.copy()
        cfg["depth_deadband"] = 0.05  # %5 ölü bölge

        # %2'lik küçük bir değişim (ölü bölge içi) -> hareket sıfır olmalı
        d_small = map_camera_deltas_to_robot(
            dx_cam=0.0, dy_cam=0.0,
            hand_width=102.0, anchor_hand_width=100.0,
            config=cfg
        )
        self.assertEqual(d_small[1], 0.0)

        # %15'lik belirgin bir yaklaşma (ölü bölge dışı) -> robot ileri (-Y) gitmeli
        d_large = map_camera_deltas_to_robot(
            dx_cam=0.0, dy_cam=0.0,
            hand_width=115.0, anchor_hand_width=100.0,
            config=cfg
        )
        self.assertLess(d_large[1], 0.0)

    def test_threaded_camera_lifecycle(self):
        """ThreadedCamera sınıfının başlatılması, kare sunması ve temiz kapanması."""
        from teleop_webcam import ThreadedCamera
        try:
            cam = ThreadedCamera(camera_id=0, width=320, height=240)
        except RuntimeError:
            self.skipTest("Kamera aygıtı mevcut değil.")

        import time
        time.sleep(0.05)
        ret, frame = cam.read_latest()
        self.assertTrue(ret)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame.shape), 3)
        cam.release()
        self.assertFalse(cam.running)


if __name__ == "__main__":
    unittest.main()
