"""
MahutKol PS4 kolu / gamepad / klavye ile canlı simülasyon teleoperasyonu (kayıt yok).

Görev ortamı (MahutKol-MultiShapeRelational-v1) SAPIEN penceresinde açılır; girdi
`mahut_kol.teleop.action_source.GamepadKeyboardActionSource` ile 6 boyutlu mutlak
`pd_joint_pos` hedefine çevrilir (5 kol eklemi + kıskaç servo açısı). Kayıt için
`scripts/record_vla_demos.py --device gamepad` kullanın (aynı eşleme).

Kontroller: sol analog taban/omuz, sağ analog dirsek/bilek, D-Pad bilek roll, R2/L2 kıskaç
kapat/aç, Üçgen/Options sıfırla. Klavye (pencere odakta): A/D taban, W/S omuz, ↑/↓ dirsek,
←/→ bilek, Q/E roll, SPACE kapat, F aç, R sıfırla.

Kullanım:
  python3 scripts/teleop_maniskill_ps4.py [--seed 0] [--fps 20]
"""

import argparse
import os
import time

os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"


ENV_ID = "MahutKol-MultiShapeRelational-v1"


def main():
    parser = argparse.ArgumentParser(description="MahutKol PS4 / Gamepad / Klavye Teleoperasyonu")
    parser.add_argument("--seed", type=int, default=None, help="ilk sahnenin tohumu")
    parser.add_argument("--fps", type=int, default=20, help="kontrol frekansı (Hz); ortam 20 Hz")
    args = parser.parse_args()

    import gymnasium as gym
    import torch

    import mahut_kol.envs  # noqa: F401
    from mahut_kol.teleop.action_source import GamepadKeyboardActionSource

    env = gym.make(ENV_ID, obs_mode="state_dict", control_mode="pd_joint_pos", render_mode="human",
                   max_episode_steps=50000)
    u = env.unwrapped
    src = GamepadKeyboardActionSource()
    qpos = lambda: u.agent.robot.get_qpos()[0].cpu().numpy()  # noqa: E731

    def reset(seed=None):
        env.reset(seed=seed)
        src.reset(qpos())
        print(f"⟳ Komut: {u.get_command()}")

    reset(args.seed)
    dt = 1.0 / args.fps
    try:
        while True:
            t0 = time.time()
            if "reset" in src.poll_events():
                reset()
                time.sleep(0.3)
                continue
            _, _, terminated, truncated, info = env.step(torch.as_tensor(src.act(qpos())))
            try:
                env.render()
            except Exception:
                print("\n[BİLGİ] Simülasyon penceresi kapatıldı.")
                break
            if bool(info["success"].item()):
                print("🎯 Görev başarılı! Sıfırlamak için Üçgen/Options ya da R.")
            if terminated.item() or truncated.item():
                reset()
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\n[ÇIKIŞ] Kullanıcı kapattı.")
    finally:
        src.close()
        env.close()


if __name__ == "__main__":
    main()
