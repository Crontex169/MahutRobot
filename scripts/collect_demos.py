#!/usr/bin/env python3
"""
ManiSkill PickCube Görevinden LeRobot Formatında Veri Kümesi Toplama Betiği.

Kullanım:
    python3 scripts/collect_demos.py [bölüm_sayısı]
"""
import os
import shutil
import sys
import gymnasium as gym
import torch

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mahut_kol.envs.pick_cube_task import scripted_actions
import mahut_kol.envs  # noqa: F401
import mahut_kol.robots  # noqa: F401

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
ROOT = os.path.expanduser("~/vla_data/pick_cube_arm")
TASK = "pick up the red cube"
FPS = 20
NAMES = ["joint0", "joint1", "joint2", "joint3", "joint4", "finger_left", "finger_right"]


def main():
    if not HAS_LEROBOT:
        print("⚠️ 'lerobot' kütüphanesi bulunamadı. Veri toplamak için lerobot ortamında çalıştırınız.")
        print("   pip install lerobot")
        sys.exit(1)

    shutil.rmtree(ROOT, ignore_errors=True)
    ds = LeRobotDataset.create(
        repo_id="local/pick_cube_arm",
        fps=FPS,
        root=ROOT,
        robot_type="mahut_kol",
        features={
            "observation.images.front": {
                "dtype": "video",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {"dtype": "float32", "shape": (7,), "names": NAMES},
            "action": {"dtype": "float32", "shape": (7,), "names": NAMES},
        },
    )

    env = gym.make(
        "MahutKol-PickCube-v1",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_backend="cpu",
        reward_mode="sparse",
    )
    kept, seed = 0, 0
    while kept < N:
        obs, _ = env.reset(seed=seed)
        seed += 1
        frames = []
        for a in scripted_actions(env.unwrapped.cube.pose.p[0].cpu().numpy()):
            frames.append(
                {
                    "observation.images.front": obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy(),
                    "observation.state": obs["agent"]["qpos"][0].cpu().numpy(),
                    "action": a,
                    "task": TASK,
                }
            )
            obs, _, _, _, info = env.step(torch.from_numpy(a))
        if not info["success"].item():
            print(f"seed {seed - 1}: başarısız, atlandı")
            continue
        for f in frames:
            ds.add_frame(f)
        ds.save_episode()
        kept += 1
        print(f"bölüm {kept}/{N} (seed {seed - 1})", flush=True)

    ds.finalize()
    env.close()
    print(f"✅ {kept} bölüm {ROOT} konumuna kaydedildi, başarı oranı {kept}/{seed}")


if __name__ == "__main__":
    main()
