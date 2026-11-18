"""
PS4 / gamepad / klavye ile HDF5 gösterim kaydedici (basit, LeRobot dışı).

VLA veri kümesi için ASIL kayıt yolu `scripts/record_vla_demos.py`'dir (LeRobot v3, komut
kuyruğu, kameralar). Bu betik yalnızca eylem/state yörüngelerini hızlı denemeler için HDF5'e yazar.

Görev ortamı MahutKol-MultiShapeRelational-v1, `pd_joint_pos`. Girdi
`GamepadKeyboardActionSource` ile 6 boyutlu mutlak hedefe çevrilir. Her adımda eylem (6) ve
`robot_state` (6: 5 kol + kıskaç servo açısı) saklanır; yalnızca ortamın başarılı saydığı
bölümler yazılır. Sıfırla (Üçgen/Options ya da R) bölümü iptal eder.

Kullanım:
  python3 Teleoperasyonlar/record_demos.py --output-path data/demos/raw_demos.h5 --num-episodes 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for _p in (os.path.join(_ROOT_DIR, "src"), _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ENV_ID = "MahutKol-MultiShapeRelational-v1"


def record_demonstrations(output_path: str, num_episodes: int = 50, control_freq: int = 20, make_source=None,
                          render: bool = True, max_steps: int = 600):
    """Teleop döngüsü; başarılı bölümleri HDF5'e (episode_<i>/actions, states) yazar.
    make_source(env) -> ActionSource (varsayılan: gamepad/klavye; testlerde scripted)."""
    import gymnasium as gym
    import h5py
    import torch

    import mahut_kol.envs  # noqa: F401
    from mahut_kol.robots.mahut_robot import robot_state
    from mahut_kol.teleop.action_source import GamepadKeyboardActionSource

    env = gym.make(ENV_ID, obs_mode="state_dict", control_mode="pd_joint_pos",
                   render_mode="human" if render else None, max_episode_steps=max_steps)
    u = env.unwrapped
    src = make_source(env) if make_source else GamepadKeyboardActionSource()
    qpos = lambda: u.agent.robot.get_qpos()[0].cpu().numpy()  # noqa: E731
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dt = 1.0 / control_freq
    saved = 0
    print(f"🎬 Kayıt: {output_path} | hedef {num_episodes} başarılı bölüm | {control_freq} Hz")

    with h5py.File(output_path, "a") as h5:
        saved = int(h5.attrs.get("total_episodes", 0))
        try:
            while saved < num_episodes:
                env.reset()
                src.reset(qpos())
                print(f"\n▶ Bölüm {saved + 1}/{num_episodes}: {u.get_command()}")
                actions, states, success, aborted = [], [], False, False
                while True:
                    t0 = time.time()
                    if "reset" in src.poll_events():
                        aborted = True
                        break
                    q = qpos()
                    a = np.asarray(src.act(q), dtype=np.float32)
                    states.append(robot_state(q).astype(np.float32))
                    actions.append(a)
                    _, _, terminated, truncated, info = env.step(torch.as_tensor(a))
                    if render:
                        env.render()
                    if bool(info["success"].item()):
                        success = True
                        break
                    if terminated.item() or truncated.item():
                        break
                    if render:
                        time.sleep(max(0.0, dt - (time.time() - t0)))
                if not success or len(actions) <= 5:
                    print("  ⟳ kaydedilmedi" + (" (iptal)" if aborted else ""))
                    continue
                g = h5.create_group(f"episode_{saved}")
                g.create_dataset("actions", data=np.stack(actions))
                g.create_dataset("states", data=np.stack(states))
                g.attrs["num_steps"] = len(actions)
                g.attrs["command"] = json.dumps(u.get_command())
                saved += 1
                h5.attrs["total_episodes"] = saved
                h5.flush()
                print(f"💾 Bölüm {saved} kaydedildi ({len(actions)} adım)")
        except KeyboardInterrupt:
            print("\n[ÇIKIŞ] Kullanıcı durdurdu.")
        finally:
            src.close()
            env.close()
    print(f"[BİLGİ] Toplam {saved} başarılı bölüm: {output_path}")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MahutKol HDF5 gösterim kaydedici (PS4/gamepad/klavye)")
    parser.add_argument("--output-path", default="data/demos/raw_demos.h5", help="HDF5 yolu (varsa sonuna eklenir)")
    parser.add_argument("--num-episodes", type=int, default=50, help="toplam başarılı bölüm hedefi")
    parser.add_argument("--freq", type=int, default=20, help="kontrol frekansı (Hz)")
    args = parser.parse_args()
    record_demonstrations(args.output_path, args.num_episodes, args.freq)
