#!/usr/bin/env python3
"""
ManiSkill 3 - MahutKol Özel Görev Görselleştirici ve Test Betiği (PushCube).

Bu betik:
1. Yeni oluşturulan özel görev ortamını ('MahutKol-PushCube-v0') başlatır.
2. SAPIEN GUI penceresinde robotu, masayı, kırmızı küpü ve hedef halkasını canlı gösterir.
3. PS4 Kolu (Gamepad) veya Klavye ile gerçek zamanlı teleoperasyon sağlar.
4. Sinüs dalgalı otomatik hareket, rastgele eylemler veya ekran görüntüsü kaydetme modlarını destekler.

Kullanım:
    python3 scripts/sim_custom_task.py                 # Varsayılan: PS4 / Klavye Teleoperasyon
    python3 scripts/sim_custom_task.py --mode sine     # Sinüs dalgalı otomatik kol hareketi
    python3 scripts/sim_custom_task.py --mode random   # Rastgele aksiyon testi
    python3 scripts/sim_custom_task.py --save-snapshot # docs/mahut_kol_push_cube.png görüntüsü al ve çık
"""

import argparse
import os
import sys
import time
import numpy as np

# src/ dizinini Python yoluna ekle
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import gymnasium as gym
import mahut_kol.envs
import mahut_kol.robots
from mahut_kol.utils.teleop_ps4 import MahutPS4Controller


def main():
    parser = argparse.ArgumentParser(description="MahutKol Özel Görev (PushCube) Görselleştirici")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["teleop", "idle", "sine", "random"],
        default="teleop",
        help="Çalışma modu: 'teleop' (Gamepad/Klavye), 'sine' (otomatik hareket), 'idle' (sabit), 'random' (rastgele)",
    )
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="Sahnenin anlık görüntüsünü 'docs/mahut_kol_push_cube.png' olarak kaydeder ve çıkar.",
    )
    args = parser.parse_args()

    env_id = "MahutKol-PushCube-v0"

    if args.save_snapshot:
        import cv2
        print(f"📷 {env_id} sahnesinin ekran görüntüsü alınıyor...")
        env = gym.make(env_id, render_mode="rgb_array")
        obs, _ = env.reset()
        img = env.render().squeeze(0).cpu().numpy()
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out_path = os.path.abspath(os.path.join(_CURRENT_DIR, "../docs/mahut_kol_push_cube.png"))
        cv2.imwrite(out_path, bgr)
        env.close()
        print(f"✅ Görüntü kaydedildi: {out_path}")
        return

    print("=" * 68)
    print(f"🤖 ManiSkill 3 Özel Görev Başlatılıyor: {env_id}")
    print("=" * 68)
    print("💡 Görev Amacı: Kırmızı küpü masa üzerindeki hedef halkasına itin.")
    print("💡 SAPIEN Görselleştirici Kontrolleri:")
    print("  - Sol Fare Tuşu: Sahneyi döndürme (Orbit)")
    print("  - Sağ Fare Tuşu / Kaydırma: Yakınlaştırma (Zoom)")
    print("  - Boşluk Tuşu (Space): Simülasyonu duraklat / devam ettir")
    print("-" * 68)

    controller = None
    if args.mode == "teleop":
        controller = MahutPS4Controller()
        has_gamepad = controller.init_joystick()
        if has_gamepad:
            print(f"🎮 PS4 / Gamepad Algılandı: {controller.get_controller_name()}")
            print("🎮 Teleoperasyon Kontrolleri:")
            print("  - Sol Analog (Yatay) : Taban Dönüşü (joint1)")
            print("  - Sol Analog (Dikey) : Omuz Açısı (joint2)")
            print("  - Sağ Analog (Dikey) : Dirsek Açısı (joint3)")
            print("  - Sağ Analog (Yatay) : Bilek Eğimi (joint4)")
            print("  - L2 / R2 Tetikleri  : Kıskaç Aç (L2) / Kapat (R2)")
            print("  - Üçgen / Options    : Ortamı Sıfırla (Reset)")
        else:
            print("⚠️ Gamepad algılanamadı! Klavye kontrolleri devrede:")
            print("  - W/S: Omuz, A/D: Taban")
            print("  - Yukarı/Aşağı: Dirsek, Sol/Sağ: Bilek")
            print("  - Space: Kıskaç Kapat, E: Kıskaç Aç, R: Sıfırla")
    print("-" * 68)
    print("  - Pencereyi Kapat veya Ctrl+C: Çıkış")
    print("=" * 68)

    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode="pd_joint_target_delta_pos",
        render_mode="human",
        max_episode_steps=100000,
    )

    obs, _ = env.reset()
    viewer = env.render()

    step_count = 0
    try:
        while True:
            if hasattr(viewer, "closed") and viewer.closed:
                break

            if args.mode == "teleop" and controller is not None:
                if controller.poll_reset():
                    print("⟳ Sıfırlama istendi, sahne yeniden başlatılıyor...")
                    obs, _ = env.reset()
                    time.sleep(0.2)
                    continue
                action = controller.poll_action()
            elif args.mode == "sine":
                t = step_count * 0.05
                # Küpe doğru uzanıp yumuşak itme hareketi
                action = np.array([
                    0.20 * np.sin(t * 0.5),   # joint0 (taban)
                    -0.10 * np.cos(t * 0.4),  # joint1 (omuz)
                    -0.10 * np.sin(t * 0.4),  # joint2 (dirsek)
                    0.10 * np.sin(t * 0.8),   # joint3 (bilek pitch)
                    0.05 * np.cos(t * 0.8),   # joint4 (bilek roll)
                    0.0,                      # finger_left
                    0.0,                      # finger_right
                ], dtype=np.float32)
            elif args.mode == "random":
                action = env.action_space.sample() * 0.15
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)

            # Target clamping: Hedef açının eklem sınırlarını aşmasını önler
            agent = env.unwrapped.agent
            if hasattr(agent, "controller") and hasattr(agent.controller, "_target_qpos"):
                qlimits = agent.robot.get_qlimits()
                agent.controller._target_qpos = torch.clamp(
                    agent.controller._target_qpos, qlimits[..., 0], qlimits[..., 1]
                )

            env.render()

            # Başarı durumu loglama
            if bool(info.get("success", False)):
                print(f"🎉 BAŞARILI! Küp hedefe ulaştı. Adım: {step_count}")

            if terminated or truncated:
                obs, _ = env.reset()

            step_count += 1
            time.sleep(0.015)

    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    finally:
        env.close()
        print("Ortam başarıyla kapatıldı.")


if __name__ == "__main__":
    main()
