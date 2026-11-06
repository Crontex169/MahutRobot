"""
MahutKol (Özel Tasarım 5-DOF Kol + Paralel Kıskaç) PS4 Kolu / Gamepad Teleoperasyon Betiği.

Bu betik:
1. MahutKol özel robot simülasyonunu SAPIEN Viewer ile açar (varsayılan: MahutKol-PickCube-v1).
2. PS4 kolunu (DualShock 4), uyumlu gamepad veya klavyeyi bağlayarak analog çubuklar ve tetiklerle robotu sürer.
3. 'pd_joint_target_delta_pos' kontrol modu ve eklem sınırlayıcı (target clamping) ile:
   - Kol küpü tuttuğunda tork sıfırlanmaz, kavrama kuvveti (holding force) korunur.
   - Kol havada veya masada asla kendi kendine kaymaz (zero drift).
   - Eklem hedefi sınırları aşmadığı için integral wind-up ve fiziksel fırlamalar engellenir.
   - Gerçekçi motor tork sınırları (FT5330M 3.432 Nm kol / 15 N kıskaç) ve yüksek kavrama sürtünmesi (mu=2.0) devrededir.

Kullanım:
  # 1. MahutKol Özel Robotu ile Çalıştırma (Varsayılan):
  python3 Teleoperasyonlar/teleop_maniskill_ps4.py

  # 2. MahutKol Diğer Görev Ortamları:
  python3 Teleoperasyonlar/teleop_maniskill_ps4.py --env-id MahutKol-PushCube-v0
  python3 Teleoperasyonlar/teleop_maniskill_ps4.py --env-id MahutKol-Minimal-v0

  # 3. İsteğe Bağlı: Alternatif Ortamlar:
  python3 Teleoperasyonlar/teleop_maniskill_ps4.py --env-id PickCubeSO100-v1
"""

import argparse
import os
import sys
import time
import numpy as np
import torch

os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

# Proje ve src/ dizinlerini Python yoluna ekle
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../src"))
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    import pygame
except ImportError:
    print("[HATA] Pygame kütüphanesi eksik! Yüklemek için: pip install pygame")
    sys.exit(1)


class PS4TeleopController:
    """ManiSkill ve MahutKol robotları için genel PS4 Kontrolcüsü."""

    def __init__(self, deadzone: float = 0.08, scale: float = 0.6):
        self.deadzone = deadzone
        self.scale = scale
        self.joystick = None

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"[BAĞLANDI] Gamepad: {self.joystick.get_name()}")
        else:
            print("[UYARI] Gamepad bulunamadı! Klavye kontrolleri devrede.")

        self.gripper_val = 0.0

    def _deadzone(self, val: float) -> float:
        if abs(val) < self.deadzone:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * (abs(val) - self.deadzone) / (1.0 - self.deadzone)

    def get_action_for_env(self, action_dim: int, is_so100: bool = False, is_mahut: bool = True) -> np.ndarray:
        """
        Ortamın eylem uzayına göre PS4 kolundan veya klavyeden aksiyon üretir.
        """
        pygame.event.pump()
        action = np.zeros(action_dim, dtype=np.float32)

        if self.joystick is not None:
            num_axes = self.joystick.get_numaxes()

            raw_lx = self.joystick.get_axis(0) if num_axes > 0 else 0.0
            raw_ly = self.joystick.get_axis(1) if num_axes > 1 else 0.0

            if num_axes >= 6:
                # Standart Linux PS4 / DualShock / Xbox eksen haritası
                raw_l2 = self.joystick.get_axis(2)
                raw_rx = self.joystick.get_axis(3)
                raw_ry = self.joystick.get_axis(4)
                raw_r2 = self.joystick.get_axis(5)

                l2 = max(0.0, (raw_l2 + 1.0) / 2.0)
                r2 = max(0.0, (raw_r2 + 1.0) / 2.0)
            else:
                raw_rx = self.joystick.get_axis(2) if num_axes > 2 else 0.0
                raw_ry = self.joystick.get_axis(3) if num_axes > 3 else 0.0
                l2 = 0.0
                r2 = 0.0

            lx = self._deadzone(raw_lx) * self.scale
            ly = self._deadzone(raw_ly) * self.scale
            rx = self._deadzone(raw_rx) * self.scale
            ry = self._deadzone(raw_ry) * self.scale

            # D-Pad şapka okuma (Bilek roll için)
            hat_x = 0.0
            if self.joystick.get_numhats() > 0:
                hat_x = float(self.joystick.get_hat(0)[0]) * self.scale

            if is_mahut or action_dim == 7:
                # MahutKol: 5-DOF Kol + 2-Parmak Tutucu
                # action[0]: Taban Yaw (Sol Stick X)
                # action[1]: Omuz Pitch (Sol Stick Y, ileri itince omuz aşağı/ileri)
                # action[2]: Dirsek Pitch (Sağ Stick Y)
                # action[3]: Bilek Pitch (Sağ Stick X)
                # action[4]: Bilek Roll (D-Pad Sol/Sağ)
                # action[5, 6]: Parmaklar (R2 basınca kapat: +delta, L2 basınca aç: -delta)
                action[0] = -lx
                action[1] = -ly
                action[2] = -ry
                action[3] = rx
                action[4] = hat_x

                # Parmak delta kontrolü (kapatmak için 0.042 yönü pozitiftir)
                grip_cmd = 0.0
                if r2 > 0.05:
                    grip_cmd = float(r2) * 0.8
                elif l2 > 0.05:
                    grip_cmd = -float(l2) * 0.8

                action[5] = grip_cmd
                action[6] = grip_cmd

            elif is_so100 or action_dim == 6:
                # SO-100: 5-DOF Kol + 1-DOF Kıskaç
                # Eklem sırası: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
                action[0] = -lx
                action[1] = -ly
                action[2] = -ry
                action[3] = rx
                action[4] = hat_x

                # SO-100 kıskacı: -1.1 kapalı, +1.1 açık
                grip_cmd = 0.0
                if r2 > 0.05:
                    grip_cmd = -float(r2) * 0.8  # Kapat
                elif l2 > 0.05:
                    grip_cmd = float(l2) * 0.8   # Aç

                action[5] = grip_cmd

            else:
                # Franka Panda Kartezyen EEF Kontrolü:
                # action[0:3]: dx, dy, dz
                # action[3:6]: droll, dpitch, dyaw
                # action[6]: gripper
                if action_dim >= 1: action[0] = -ly
                if action_dim >= 2: action[1] = lx
                if action_dim >= 3: action[2] = -ry
                if action_dim >= 6: action[5] = rx
                grip_cmd = 0.0
                if r2 > 0.05: grip_cmd = -float(r2)
                elif l2 > 0.05: grip_cmd = float(l2)
                if action_dim >= 7: action[6] = grip_cmd

        # Klavye Yedek Kontrolleri
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] and action_dim > 0: action[0] = self.scale
        if keys[pygame.K_d] and action_dim > 0: action[0] = -self.scale
        if keys[pygame.K_w] and action_dim > 1: action[1] = self.scale
        if keys[pygame.K_s] and action_dim > 1: action[1] = -self.scale
        if keys[pygame.K_UP] and action_dim > 2: action[2] = self.scale
        if keys[pygame.K_DOWN] and action_dim > 2: action[2] = -self.scale
        if keys[pygame.K_LEFT] and action_dim > 3: action[3] = -self.scale
        if keys[pygame.K_RIGHT] and action_dim > 3: action[3] = self.scale
        if keys[pygame.K_q] and action_dim > 4: action[4] = -self.scale
        if keys[pygame.K_e] and action_dim > 4: action[4] = self.scale

        if action_dim == 7:
            # MahutKol parmaklar
            if keys[pygame.K_SPACE]:
                action[5] = 0.8
                action[6] = 0.8
            elif keys[pygame.K_f]:
                action[5] = -0.8
                action[6] = -0.8
        elif action_dim == 6:
            # SO-100 kıskaç
            if keys[pygame.K_SPACE]:
                action[5] = -0.8
            elif keys[pygame.K_f]:
                action[5] = 0.8

        return action

    def is_reset_pressed(self) -> bool:
        """Üçgen (Button 2) veya Options (Button 9) butonuna basıldıysa True."""
        pygame.event.pump()
        if self.joystick is not None:
            num_buttons = self.joystick.get_numbuttons()
            if num_buttons > 9:
                if self.joystick.get_button(2) or self.joystick.get_button(9):
                    return True
            elif num_buttons > 3:
                if self.joystick.get_button(2) or self.joystick.get_button(3):
                    return True
        keys = pygame.key.get_pressed()
        return bool(keys[pygame.K_r])


def configure_gripper_safety(env, is_so100: bool = False, is_mahut: bool = False):
    """
    Fizik motorunda tutucu eklemlerinin gerçekçi tork limitlerini ayarlar ve
    küp üzerine yüksek sürtünmeli temas materyali tanımlayarak stabil kavrama sağlar.
    """
    try:
        import sapien
        agent = env.unwrapped.agent

        if is_so100:
            # SO-100: STS3215 servo için 4.0 Nm tork limiti ve dengeli sertlik
            for joint in agent.robot.get_active_joints():
                if "gripper" in joint.name.lower():
                    joint.set_drive_properties(stiffness=200.0, damping=20.0, force_limit=4.0)
            if hasattr(agent, "controller") and hasattr(agent.controller, "config"):
                cfg = agent.controller.config
                n = len(agent.controller.joints)
                fl = [100.0] * (n - 1) + [4.0] if not isinstance(cfg.force_limit, list) else cfg.force_limit
                fl[-1] = 4.0
                cfg.force_limit = fl
                agent.controller.set_drive_property()
            print("[BİLGİ] SO-100 kıskaç tork limiti 4.0 N·m olarak ayarlandı.")

        elif is_mahut:
            # MahutKol: FT5330M (3.432 Nm kol eklemleri, 15 N kıskaç parmakları)
            for joint in agent.robot.get_active_joints():
                if "finger" in joint.name.lower():
                    joint.set_drive_properties(stiffness=150.0, damping=15.0, force_limit=15.0)
                else:
                    joint.set_drive_properties(stiffness=80.0, damping=10.0, force_limit=3.432)
            print("[BİLGİ] MahutKol FT5330M tork limitleri (Kol: 3.432 Nm, Kıskaç: 15 N) doğrulandı.")

        # Sahnedeki küpe yüksek statik ve dinamik sürtünme tanımla (silikon/kauçuk kavrama)
        if hasattr(env.unwrapped, "cube"):
            cube = env.unwrapped.cube
            mat = sapien.physx.PhysxMaterial(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
            for obj in cube._objs:
                comp = obj.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
                if comp is not None:
                    for shape in comp.collision_shapes:
                        shape.set_physical_material(mat)
            print("[BİLGİ] Küp yüzeyine yüksek sürtünmeli kavrama materyali (mu=2.0) tanımlandı.")

    except Exception as e:
        print(f"[UYARI] configure_gripper_safety yapılandırılamadı: {e}")


def main():
    parser = argparse.ArgumentParser(description="MahutKol PS4 / Gamepad Teleoperasyon Denetleyicisi")
    parser.add_argument(
        "--env-id",
        type=str,
        default="MahutKol-PickCube-v1",
        help="ManiSkill ortamı ('MahutKol-PickCube-v1', 'MahutKol-PushCube-v0', 'MahutKol-Minimal-v0', 'PickCubeSO100-v1')",
    )
    parser.add_argument("--fps", type=int, default=30, help="Simülasyon kare hızı (Hz)")
    args = parser.parse_args()

    # ManiSkill & MahutKol İçe Aktarma Kontrolü
    try:
        import gymnasium as gym
        import mani_skill.envs
        import mahut_kol.envs
        import mahut_kol.robots
    except ImportError as e:
        print("=" * 70)
        print("⚠️ [UYARI] Gerekli kütüphaneler yüklenirken hata oluştu!")
        print("Hata detayı:", e)
        sys.exit(1)

    print("=" * 60)
    print(f"🤖 ManiSkill Simülasyonu Başlatılıyor: {args.env_id}")
    print("=" * 60)

    # Ortam Kontrol Modu Belirleme:
    # 'pd_joint_target_delta_pos' modu kullanılarak:
    # 1. Analog veya tetik bırakıldığında (delta=0) hedef konum sabit tutulur.
    # 2. Motorlar tam torkla küpü kavramaya ve kolu havada tutmaya devam eder (zero drift).
    is_so100 = "SO100" in args.env_id
    is_mahut = ("Mahut" in args.env_id or "Arm" in args.env_id) or not is_so100
    control_mode = "pd_joint_target_delta_pos" if (is_so100 or is_mahut) else "pd_ee_delta_pose"

    make_kwargs = {
        "obs_mode": "rgbd",
        "render_mode": "human",
        "max_episode_steps": 50000,
    }

    try:
        env = gym.make(
            args.env_id,
            control_mode=control_mode,
            **make_kwargs,
        )
    except Exception as e:
        print(f"[UYARI] '{control_mode}' kontrol modu açılamadı, varsayılan mod deneniyor: {e}")
        env = gym.make(args.env_id, **make_kwargs)

    action_dim = env.action_space.shape[0]
    if action_dim == 7:
        is_mahut = True
        is_so100 = False
    elif action_dim == 6 and is_so100:
        is_mahut = False

    print(f"[BİLGİ] Robot Modeli            : {'MahutKol Özel Model (5-DOF + 2 Parmak Tutucu)' if is_mahut else ('SO-100 (5-DOF + 1 Kıskaç)' if is_so100 else 'Diğer Robot')}")
    print(f"[BİLGİ] Eylem Boyutu (Action Dim): {action_dim}")
    print(f"[BİLGİ] Kontrol Modu           : {getattr(env.unwrapped, 'control_mode', 'default')}")
    print("-" * 60)
    print("🎮 Kontroller:")
    print("  - Sol Analog   : Gövde Taban Dönüşü & Omuz Hareketi")
    print("  - Sağ Analog   : Dirsek Hareketi & Bilek Eğimi")
    print("  - D-Pad        : Bilek Yuvarlanma (Roll)")
    print("  - R2 / L2      : Kıskaç Kapat (R2) / Aç (L2)")
    print("  - Üçgen / Opt  : Ortamı Sıfırla (Reset)")
    print("  - Klavye       : W/S (Omuz), A/D (Taban), Oklar (Dirsek/Bilek), Space (Kapat), F (Aç)")
    print("  - Çıkış        : Ctrl+C")
    print("-" * 60)

    controller = PS4TeleopController()
    obs, info = env.reset()
    configure_gripper_safety(env, is_so100=is_so100, is_mahut=is_mahut)
    dt = 1.0 / args.fps

    try:
        while True:
            step_start = time.time()

            if controller.is_reset_pressed():
                print("⟳ Ortam sıfırlandı...")
                obs, info = env.reset()
                configure_gripper_safety(env, is_so100=is_so100, is_mahut=is_mahut)
                time.sleep(0.3)
                continue

            action = controller.get_action_for_env(action_dim, is_so100=is_so100, is_mahut=is_mahut)
            obs, reward, terminated, truncated, info = env.step(action)

            # Target qpos Clamping:
            # Hedef pozisyonun eklem limitlerinin dışına kaçmasını (wind-up patlaması) engeller
            agent = env.unwrapped.agent
            if hasattr(agent, "controller") and hasattr(agent.controller, "_target_qpos"):
                qlimits = agent.robot.get_qlimits()
                agent.controller._target_qpos = torch.clamp(
                    agent.controller._target_qpos, qlimits[..., 0], qlimits[..., 1]
                )

            try:
                env.render()
            except (AttributeError, Exception):
                print("\n[BİLGİ] Simülasyon penceresi kapatıldı.")
                break

            if terminated or truncated:
                print("⟳ Bölüm süresi doldu veya tamamlandı, ortam sıfırlanıyor...")
                obs, info = env.reset()
                configure_gripper_safety(env, is_so100=is_so100, is_mahut=is_mahut)
                time.sleep(0.3)
                continue

            if bool(info.get("success", False)):
                print("🎯 TEBRİKLER! Görev Başarılı (Küp yakalandı / hedefe ulaştı)!")

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[ÇIKIŞ] Simülasyon kullanıcı tarafından kapatıldı.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
