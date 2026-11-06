#!/usr/bin/env python3
"""
ManiSkill 3 - MahutKol Sade Ortam Görselleştirici ve PS4 Teleoperasyon Betiği.

Bu betik:
1. En sade masaüstü ortamını ('MahutKol-Minimal-v0') başlatır.
2. SAPIEN GUI interaktif penceresini ('render_mode="human"') açarak robotu sahnede gösterir.
3. PS4 kolu (DualShock 4) ve uyumlu gamepad'ler ile gerçek zamanlı teleoperasyon sağlar.
4. Fare ile 360° sahne döndürme/yakınlaştırma destekler.

Kullanım:
    python3 scripts/sim_minimal.py                 # Varsayılan: PS4 Teleoperasyon modu
    python3 scripts/sim_minimal.py --mode teleop   # PS4 Teleoperasyon modu
    python3 scripts/sim_minimal.py --mode sine     # Sinüs dalgalı otomatik eklem testi
    python3 scripts/sim_minimal.py --mode idle     # Sabit duruş
    python3 scripts/sim_minimal.py --mode random   # Rastgele aksiyon testi
"""

import argparse
import os
import sys
import time
import numpy as np

# src/ dizinini otomatik olarak Python arama yoluna (sys.path) ekle
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Yerel modülleri dahil et
import gymnasium as gym
import mahut_kol.envs
import mahut_kol.robots
from mahut_kol.utils.teleop_ps4 import MahutPS4Controller


def main():
    parser = argparse.ArgumentParser(description="MahutKol Sade ManiSkill Görselleştirici ve PS4 Teleoperasyon")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["teleop", "idle", "sine", "random"],
        default="teleop",
        help="Hareket modu: 'teleop' (PS4/Gamepad kontrolü), 'idle' (sabit), 'sine' (sinüs testi), 'random' (rastgele)",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("🤖 ManiSkill 3 Sade Sahne Başlatılıyor: MahutKol-Minimal-v0")
    print("=" * 65)
    print("💡 SAPIEN Görselleştirici Kontrolleri:")
    print("  - Sol Fare Tuşu: Sahneyi döndürme (Orbit)")
    print("  - Sağ Fare Tuşu / Kaydırma: Yakınlaştırma (Zoom)")
    print("  - Boşluk Tuşu (Space): Simülasyonu duraklat / devam ettir")
    print("-" * 65)

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
            print("⚠️ Gamepad algılanamadı! Klavye yedek kontrolleri devrede:")
            print("  - W/S: Omuz, A/D: Taban")
            print("  - Yukarı/Aşağı: Dirsek, Sol/Sağ: Bilek")
            print("  - Space: Kıskaç Kapat, E: Kıskaç Aç, R: Sıfırla")
    print("-" * 65)
    print("  - Pencereyi Kapat veya Ctrl+C: Çıkış")
    print("=" * 65)

    env = gym.make(
        "MahutKol-Minimal-v0",
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
            # SAPIEN 3 viewer durumu
            if hasattr(viewer, "closed") and viewer.closed:
                break

            if args.mode == "teleop" and controller is not None:
                if controller.poll_reset():
                    print("⟳ Sıfırlama istendi, robot başlangıç pozisyonuna dönüyor...")
                    obs, _ = env.reset()
                    time.sleep(0.2)
                    continue
                action = controller.poll_action()
            elif args.mode == "sine":
                t = step_count * 0.05
                # Kol eklemlerine yumuşak salınım, kıskaca açma-kapama
                action = np.array([
                    0.25 * np.sin(t),        # joint0 (taban yaw)
                    0.20 * np.cos(t * 0.8),  # joint1 (omuz)
                    0.20 * np.sin(t * 0.8),  # joint2 (dirsek)
                    0.15 * np.sin(t * 1.2),  # joint3 (bilek pitch)
                    0.10 * np.cos(t * 1.0),  # joint4 (bilek roll)
                    0.30 * np.sin(t * 0.5),  # finger_left
                    0.30 * np.sin(t * 0.5),  # finger_right
                ], dtype=np.float32)
            elif args.mode == "random":
                action = env.action_space.sample() * 0.15
            else:
                action = np.zeros(env.action_space.shape, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

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
