#!/usr/bin/env python3
"""
Eğitilmiş LeRobot politikasını MahutKol-PickCube-v1 ortamında değerlendirir ve başarı oranını ölçer.

Kullanım:
    python3 scripts/eval_policy.py <checkpoint_yolu> [bölüm_sayısı] [talimat]
    python3 scripts/eval_policy.py latest 5
"""
import os
import sys
import gymnasium as gym
import torch

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import mahut_kol.envs  # noqa: F401
import mahut_kol.robots  # noqa: F401

try:
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from mani_skill.utils.wrappers.record import RecordEpisode
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

if len(sys.argv) < 2:
    print("Kullanım: python3 scripts/eval_policy.py <checkpoint>/pretrained_model [n_episodes] [instruction]")
    sys.exit(0)

CKPT = sys.argv[1]
if CKPT == "latest":
    CKPT = os.path.expanduser("~/vla_runs/smolvla_mahut_kol/checkpoints/last/pretrained_model")
CKPT = os.path.realpath(CKPT)
STEP = os.path.basename(os.path.dirname(CKPT))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
TASK = sys.argv[3] if len(sys.argv) > 3 else "pick up the red cube"
RENAME = {"observation.images.front": "observation.images.camera1"}


def main():
    if not HAS_LIBS:
        print("⚠️ lerobot veya record wrapper bulunamadı.")
        sys.exit(1)

    cfg = PreTrainedConfig.from_pretrained(CKPT)
    cfg.device = os.environ.get("EVAL_DEVICE", cfg.device)
    policy = get_policy_class(cfg.type).from_pretrained(CKPT, config=cfg).to(cfg.device).eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=CKPT,
        preprocessor_overrides={
            "device_processor": {"device": str(cfg.device)},
            "rename_observations_processor": {"rename_map": RENAME},
        },
    )

    env = gym.make(
        "MahutKol-PickCube-v1",
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        render_backend="cpu",
        reward_mode="sparse",
    )
    env = RecordEpisode(env, f"videos_eval/{STEP}", save_trajectory=False, video_fps=20, max_steps_per_video=None)
    print(f"Checkpoint adımı: {STEP}, cihaz: {cfg.device}")
    wins = 0
    for ep in range(N):
        obs, _ = env.reset(seed=10000 + ep)
        policy.reset()
        success = False
        for _ in range(200):
            img = obs["sensor_data"]["base_camera"]["rgb"].permute(0, 3, 1, 2).float() / 255
            batch = pre(
                {
                    "observation.images.front": img,
                    "observation.state": obs["agent"]["qpos"][:, :6].float(),
                    "task": [TASK],
                }
            )
            with torch.inference_mode():
                action = post(policy.select_action(batch))
            obs, _, _, _, info = env.step(action.cpu())
            if info["success"].item():
                success = True
                break
        wins += success
        print(f"Bölüm {ep}: {'BAŞARILI' if success else 'BAŞARISIZ'}  küp x={env.unwrapped.cube.pose.p[0, 0].item():.3f}", flush=True)
    env.close()
    print(f"Başarı oranı: {wins}/{N} (Görev: '{TASK}')")


if __name__ == "__main__":
    main()
