"""
Aşama 3: telefon (WebXR) teleop - telefonsuz testler.
- Sahte WebSocket mesajları (FastAPI TestClient) -> paylaşılan durum
- Eksen eşlemesi: telefon ileri -> TCP -x, sağ -> +y, yukarı -> +z
- Debriyaj bırakılınca kol sabit; kıskaç kaydırıcısı -> parmak eklemi
- IK çalışma alanındaki hedeflere ulaşıyor (sayısal)
- Uçtan uca: sahte mesajlarla ManiSkill env.step döngüsünde TCP hedefe gidiyor
"""

import json
import unittest
import warnings

import numpy as np

from mahut_kol.envs.multi_shape_task import (
    ROBOT_REST_QPOS,
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
)
from mahut_kol.robots.kinematics import get_kinematics
from mahut_kol.teleop.action_source import FINGER_CLOSED, ScriptedActionSource, clip_action
from mahut_kol.teleop.phone_webxr import (
    DEFAULT_HTML,
    PhoneActionSource,
    PhoneTeleopMapper,
    TeleopSharedState,
    create_app,
    phone_to_world_delta,
)


def teleop_msg(dpos=(0, 0, 0), deul=(0, 0, 0), clutch=True, gripper=1.0):
    return json.dumps({"type": "teleop", "clutch": clutch, "delta_pos": list(dpos),
                       "delta_euler": list(deul), "gripper": gripper})


class TestProtocol(unittest.TestCase):
    def test_websocket_fake_messages(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from fastapi.testclient import TestClient
        state = TeleopSharedState()
        client = TestClient(create_app(state, DEFAULT_HTML))
        r = client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("WebSocket", r.text)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(teleop_msg((0.01, 0.02, -0.03), (0.1, 0, 0.2), True, 0.25))
            ws.send_text("bozuk json {")
            ws.send_text(json.dumps({"type": "reset"}))
            state.push_status({"type": "status", "instruction": "put the blue cube behind the red cube"})
            msg = json.loads(ws.receive_text())
            self.assertEqual(msg["instruction"], "put the blue cube behind the red cube")
        clutch, pressed, dpos, deul, grip, reset = state.get_state()
        self.assertFalse(clutch)  # bağlantı kopunca debriyaj bırakılır
        np.testing.assert_allclose(dpos, [0.01, 0.02, -0.03])
        np.testing.assert_allclose(deul, [0.1, 0, 0.2])
        self.assertAlmostEqual(grip, 0.25)
        self.assertTrue(reset)
        self.assertEqual(state.packet_count, 1)

    def test_clutch_edge_and_timeout(self):
        s = TeleopSharedState()
        s.handle_message(teleop_msg(clutch=True))
        self.assertEqual(s.get_state()[:2], (True, True))
        self.assertEqual(s.get_state()[:2], (True, False))
        s.last_packet_time -= 10
        self.assertFalse(s.get_state(timeout_s=0.5)[0])

    def test_axis_mapping(self):
        # telefon ileri (−Z) -> dünya −x ; sağ (+X) -> +y ; yukarı (+Y) -> +z
        np.testing.assert_allclose(phone_to_world_delta([0, 0, -0.05]), [-0.05, 0, 0])
        np.testing.assert_allclose(phone_to_world_delta([0.05, 0, 0]), [0, 0.05, 0])
        np.testing.assert_allclose(phone_to_world_delta([0, 0.05, 0]), [0, 0, 0.05])


class TestMapper(unittest.TestCase):
    def setUp(self):
        self.K = get_kinematics()
        self.m = PhoneTeleopMapper(config={"ema_alpha_pos": 1.0, "ema_alpha_rot": 1.0, "control_frame": "world"})
        self.m.reset(ROBOT_REST_QPOS[:5])
        self.p0, _ = self.K.fk(self.m.q_cmd)

    def _run(self, dpos, n=40, deul=(0, 0, 0)):
        """Telefon sayfası gibi: debriyaja basılınca delta 0, sonra hedef delta."""
        q = self.m.step(True, True, (0, 0, 0), (0, 0, 0))
        for _ in range(n):
            q = self.m.step(True, False, dpos, deul)
        return q

    def test_directions(self):
        for dphone, axis, sign in [((0, 0, -0.04), 0, -1), ((0.04, 0, 0), 1, 1), ((0, 0.04, 0), 2, 1)]:
            self.m.reset(ROBOT_REST_QPOS[:5])
            q = self._run(dphone)
            p, _ = self.K.fk(q)
            d = p - self.p0
            self.assertGreater(sign * d[axis], 0.03, (dphone, d))
            others = [i for i in range(3) if i != axis]
            self.assertLess(np.abs(d[others]).max(), 0.01, (dphone, d))

    def test_clutch_release_holds(self):
        q1 = self._run((0.03, 0, 0), n=20)
        q2 = self.m.step(False, False, (0.2, 0.2, 0.2), (0, 0, 0))
        np.testing.assert_allclose(q1, q2)
        # yeniden basınca yeni çapa: basıldığı andaki (eski) delta ile sıçrama olmaz
        q3 = self.m.step(True, True, (0.2, 0.2, 0.2), (0, 0, 0))
        np.testing.assert_allclose(q3, q2, atol=1e-6)
        # çapadan sonraki değişim yine izlenir
        q4 = self._run((0.03, 0, 0), n=20)
        self.assertGreater(self.K.fk(q4)[0][1] - self.K.fk(q3)[0][1], 0.02)

    def test_tilt_and_roll(self):
        t0 = self.K.tilt_of(self.m.q_cmd)
        q = self._run((0, 0, 0), deul=(-0.2, 0, 0.3), n=40)
        self.assertLess(self.K.tilt_of(q), t0 - 0.1)  # burun aşağı -> daha dik
        self.assertAlmostEqual(q[4], -0.3, places=2)

    def test_table_never_penetrated(self):
        from mahut_kol.robots.kinematics import finger_min_z

        q = self._run((0, -0.5, 0), n=60)  # aşağı doğru zorla
        self.assertGreaterEqual(finger_min_z(self.K, q), 0.0)

    def test_can_lift_after_touching_table(self):
        """Masaya dayanıp kalan kol (gerçek denemedeki kilitlenme) yukarı çıkabilmeli."""
        from mahut_kol.robots.kinematics import finger_min_z

        self._run((0, -0.5, 0), n=60)
        z_low = self.K.fk(self.m.q_cmd)[0][2]
        # Yerinde dururken ve aşağı zorlarken kilitlenmesin; sonra yukarı kaldır
        q = self._run((0, 0.10, 0), n=60)
        self.assertGreater(self.K.fk(q)[0][2], z_low + 0.05)
        self.assertGreaterEqual(finger_min_z(self.K, q), 0.0)

    def test_ik_reaches_workspace_targets(self):
        """Çalışma alanının üstündeki ızgarada TCP hedefe < 5 mm ulaşır."""
        errs = []
        for x in np.linspace(WORKSPACE_X_MIN, WORKSPACE_X_MAX, 4):
            for y in np.linspace(WORKSPACE_Y_MIN, WORKSPACE_Y_MAX, 4):
                for z in (0.07, 0.11):
                    self.m.reset(ROBOT_REST_QPOS[:5])
                    goal = np.array([x, y, z])
                    q = self._run(_world_to_phone(goal - self.p0), n=80)
                    p, _ = self.K.fk(q)
                    errs.append(np.linalg.norm(p - goal))
        self.assertLess(np.max(errs), 0.005, np.round(errs, 4))


def _world_to_phone(dw):
    # phone_to_world_delta'nın tersi (varsayılan eşleme)
    return np.array([dw[1], dw[2], dw[0]])


class TestPhoneActionSourceInSim(unittest.TestCase):
    def test_end_to_end_fake_phone(self):
        import gymnasium as gym
        import torch

        import mahut_kol.envs  # noqa: F401
        import mahut_kol.robots  # noqa: F401

        env = gym.make("MahutKol-MultiShapeRelational-v1", obs_mode="state_dict", control_mode="pd_joint_pos")
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        src = PhoneActionSource(start_server=False, config={"ema_alpha_pos": 1.0, "control_frame": "world"})
        qpos = u.agent.robot.get_qpos()[0].numpy()
        src.reset(qpos)
        p0 = u.agent.tcp.pose.p[0].numpy().copy()
        # Debriyaja bas (delta 0), sonra 4 cm yukarı ve 3 cm sağa, kıskacı kapat
        src.state.handle_message(teleop_msg((0, 0, 0), clutch=True, gripper=1.0))
        src.act(qpos)
        for i in range(60):
            src.state.handle_message(teleop_msg((0.03, 0.04, 0.0), clutch=True, gripper=0.0))
            a = src.act(u.agent.robot.get_qpos()[0].numpy())
            env.step(torch.as_tensor(a))
        p1 = u.agent.tcp.pose.p[0].numpy()
        d = p1 - p0
        self.assertGreater(d[1], 0.02, d)
        self.assertGreater(d[2], 0.03, d)
        self.assertAlmostEqual(float(a[5]), FINGER_CLOSED, places=5)
        src.state.handle_message(json.dumps({"type": "reset"}))
        src.act(u.agent.robot.get_qpos()[0].numpy())
        self.assertIn("reset", src.poll_events())
        env.close()

    def test_scripted_source_clip(self):
        s = ScriptedActionSource([[9, 9, 9, 9, 9, 1, -1]])
        a = s.act(np.zeros(7))
        np.testing.assert_allclose(a, clip_action([9, 9, 9, 9, 9, 1, -1]))
        self.assertTrue(s.done)


if __name__ == "__main__":
    unittest.main()


class TestWaitForStart(unittest.TestCase):
    """Bölüm, telefonda ilk debriyaja basılana dek başlamaz (süre bağlanırken dolmasın)."""

    def _src(self):
        src = PhoneActionSource(start_server=False, config={"debug_period_s": 0})
        src.reset(np.r_[ROBOT_REST_QPOS[:5], 0.0, 0.0])
        return src

    def test_waits_until_clutch(self):
        from mahut_kol.recording.session import wait_for_start

        src = self._src()
        self.assertFalse(src.ready())
        self.assertEqual(wait_for_start(None, src, poll_s=0.01, timeout_s=0.05), "timeout")
        src.state.handle_message(teleop_msg(clutch=True))
        self.assertTrue(src.ready())
        self.assertEqual(wait_for_start(None, src, poll_s=0.01, timeout_s=0.05), "start")
        # ready() kenar algılamayı tüketmemeli: ilk act() hâlâ "yeni basıldı" görüp çapalar
        clutch, pressed, *_ = src.state.get_state(src.cfg["packet_timeout_s"])
        self.assertTrue(clutch and pressed)

    def test_stale_clutch_is_not_ready(self):
        src = self._src()
        src.state.handle_message(teleop_msg(clutch=True))
        src.state.last_packet_time -= 5.0  # telefon dondu
        self.assertFalse(src.ready())

    def test_phone_reset_before_start(self):
        from mahut_kol.recording.session import wait_for_start

        src = self._src()
        src.state.handle_message(json.dumps({"type": "reset"}))
        self.assertEqual(wait_for_start(None, src, poll_s=0.01, timeout_s=0.5), "reset")

    def test_keyboard_source_always_ready(self):
        self.assertTrue(ScriptedActionSource([[0] * 7]).ready())


class TestQuaternionDelta(unittest.TestCase):
    """Açı farkı ham kuaterniyondan: telefon neredeyse yatayken de sıçramasız."""

    @staticmethod
    def _q(axis, ang):
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        return np.r_[axis * np.sin(ang / 2), np.cos(ang / 2)]

    @staticmethod
    def _mul(a, b):  # [x, y, z, w]
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])

    def test_small_rotation_near_gimbal(self):
        from mahut_kol.teleop.phone_webxr import quat_rel_rotvec

        anchor = self._q([1, 0, 0], np.radians(-85))  # telefon neredeyse yatay
        for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            cur = self._mul(self._q(axis, np.radians(10)), anchor)
            rv = quat_rel_rotvec(cur, anchor)
            np.testing.assert_allclose(rv, np.radians(10) * np.asarray(axis, float), atol=1e-9)

    def test_state_uses_raw_quat_with_own_anchor(self):
        st = TeleopSharedState()
        anchor = self._q([1, 0, 0], np.radians(-85))
        msg = lambda q: json.dumps({"type": "teleop", "clutch": True, "delta_pos": [0, 0, 0],
                                    "delta_euler": [9, 9, 9], "raw_quat": list(q), "gripper": 1})
        st.handle_message(msg(anchor))
        np.testing.assert_allclose(st.delta_euler, 0, atol=1e-12)
        st.handle_message(msg(self._mul(self._q([0, 0, 1], 0.2), anchor)))
        np.testing.assert_allclose(st.delta_euler, [0, 0, 0.2], atol=1e-9)


class TestWristCameraFrame(unittest.TestCase):
    """Varsayılan kontrol bilek kamerası görüntüsüne göre: telefon sağa -> görüntüde sağa."""

    def test_basis_matches_sim_camera_and_motion(self):
        import gymnasium as gym
        import torch
        from scipy.spatial.transform import Rotation

        import mahut_kol.envs  # noqa: F401
        import mahut_kol.robots  # noqa: F401
        from mahut_kol.teleop.phone_webxr import wrist_camera_rotation

        env = gym.make("MahutKol-MultiShapeRelational-v1", obs_mode="rgb", control_mode="pd_joint_pos")
        env.reset(seed=0)
        u = env.unwrapped
        K = get_kinematics()
        cam = u._sensors["wrist"].camera
        for q_arm in (ROBOT_REST_QPOS[:5], np.array([0.6, 0.7, 0.5, -1.6, 0.5])):
            a = torch.tensor(np.r_[q_arm, 0, 0], dtype=torch.float32)
            for _ in range(40):
                env.step(a)
            q = u.agent.robot.get_qpos()[0].numpy()
            w, x, y, z = cam.global_pose.q[0].numpy()
            Rc_sim = Rotation.from_quat([x, y, z, w]).as_matrix()
            np.testing.assert_allclose(wrist_camera_rotation(K.fk(q[:5])[1]), Rc_sim, atol=1e-4)
            img_right, img_up = -Rc_sim[:, 1], Rc_sim[:, 2]
            for dphone, axis_vec in (((0.04, 0, 0), img_right), ((0, 0, -0.04), img_up), ((0, 0.04, 0), np.array([0, 0, 1.0]))):
                m = PhoneTeleopMapper(config={"ema_alpha_pos": 1.0, "ema_alpha_rot": 1.0})
                m.reset(q[:5])
                p0 = K.fk(m.q_cmd)[0]
                m.step(True, True, (0, 0, 0), (0, 0, 0))
                for _ in range(30):
                    qq = m.step(True, False, dphone, (0, 0, 0))
                d = K.fk(qq)[0] - p0
                self.assertGreater(float(d @ axis_vec), 0.02, (q_arm, dphone, d))
        env.close()


class TestTrackingJump(unittest.TestCase):
    def test_arcore_jump_is_ignored(self):
        st = TeleopSharedState(max_packet_jump_m=0.08)
        for d in (0.0, 0.01, 0.02):
            st.handle_message(teleop_msg((d, 0, 0)))
        st.handle_message(teleop_msg((4.10, 0.8, -7.5)))   # takip kaybı: metrelerce sıçrama
        np.testing.assert_allclose(st.delta_pos, (0.02, 0, 0), atol=1e-9)
        st.handle_message(teleop_msg((4.12, 0.8, -7.5)))   # sonrasında hareket kaldığı yerden
        np.testing.assert_allclose(st.delta_pos, (0.04, 0, 0), atol=1e-9)
        self.assertEqual(st.jump_count, 1)
        st.handle_message(teleop_msg(clutch=False))
        st.handle_message(teleop_msg((0.0, 0, 0)))          # yeni vuruş: ofset sıfırlanır
        np.testing.assert_allclose(st.delta_pos, 0, atol=1e-12)


class TestFrameDriftWarning(unittest.TestCase):
    """Çerçeve debriyajda sabit; uzun yana harekette kamera dönünce operatör uyarılır."""

    def test_warns_after_long_sideways_stroke_and_clears_on_repress(self):
        K = get_kinematics()
        src = PhoneActionSource(start_server=False, config={"ema_alpha_pos": 1.0})
        q0, ok, _ = K.ik(np.array([0.17, 0.08, 0.09]), tilt=np.radians(50))
        self.assertTrue(ok)
        qpos = np.r_[q0, 0.0, 0.0]
        src.reset(qpos)
        src.state.handle_message(teleop_msg((0, 0, 0)))
        src.act(qpos)
        self.assertNotIn("YÖN KAYDI", src.status_text())
        x = 0.0
        for _ in range(120):  # telefon yavaşça 24 cm sağa: kol tabanda döner
            x += 0.002
            src.state.handle_message(teleop_msg((x, 0, 0)))
            qpos = src.act(qpos)
        self.assertGreater(src.mapper.frame_drift_deg, 30.0)
        self.assertIn("YÖN KAYDI", src.status_text())
        src.state.handle_message(teleop_msg(clutch=False))
        src.act(qpos)
        src.state.handle_message(teleop_msg((0, 0, 0)))  # yeniden bas -> çerçeve yenilenir
        src.act(qpos)
        self.assertLess(src.mapper.frame_drift_deg, 5.0)
        self.assertNotIn("YÖN KAYDI", src.status_text())
