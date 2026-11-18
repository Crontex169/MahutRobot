#!/usr/bin/env python3
"""
G2 kapısı: robot kontrolcüsü doğrulaması.

1. Adım cevabı: her kol eklemine dinlenme duruşundan +0.3 rad adım komutu; 20 Hz kontrol
   adımında %90'a varış süresi, aşım ve 1 sn sonundaki kalıcı hata.
2. Kavrama: her şekil için nesne kavrama noktasına konur, kıskaç kapatılır, kol 6 cm
   kaldırılır; nesne kıskaçla birlikte yükseldi mi ve parmaklar temas ediyor mu.

Kullanım: python3 scripts/check_robot_control.py
"""

import sys

import gymnasium as gym
import numpy as np
import torch

import mahut_kol.envs  # noqa: F401
from mahut_kol.envs.multi_shape_task import ROBOT_REST_QPOS, TABLETOP_HALF_HEIGHT, arm_heading
from mahut_kol.robots.kinematics import solve_grasp_ik
from mahut_kol.robots.mahut_robot import GRIPPER_CLOSED, robot_state

STEP = 0.3


def step_response(env):
    u = env.unwrapped
    ok = True
    print("Adım cevabı (+0.3 rad, 20 Hz):  eklem | %90 süresi | aşım | 1 sn hata")
    for j in range(5):
        env.reset(seed=0)
        a = robot_state(np.asarray(ROBOT_REST_QPOS, np.float32))
        a[j] = np.clip(a[j] + STEP if a[j] < 1.2 else a[j] - STEP, -1.5, 1.5)
        q0 = u.agent.robot.get_qpos()[0, j].item()
        traj = []
        for _ in range(20):
            env.step(torch.as_tensor(a))
            traj.append(u.agent.robot.get_qpos()[0, j].item())
        traj = (np.array(traj) - q0) / (a[j] - q0)
        t90 = (np.argmax(traj >= 0.9) + 1) / 20 if (traj >= 0.9).any() else float("inf")
        err = abs(1 - traj[-1]) * abs(a[j] - q0)
        print(f"  joint{j} | {t90:.2f} s | {max(0, traj.max() - 1) * 100:.1f}% | {np.degrees(err):.2f}°")
        ok = ok and t90 <= 0.5 and np.degrees(err) < 1.0
    return ok


def grasp(env, shape, seed):
    u = env.unwrapped
    env.reset(seed=seed, options=dict(
        manipulated_object=f"red_{shape}", reference_object="blue_cube", relation="FRONT", num_distractors=0,
        object_poses={f"red_{shape}": (0.16, 0.0, 0.0), "blue_cube": (0.20, 0.08, 0.0)},
    ))
    obj = u.manipulation_objects[f"red_{shape}"]
    x, y = 0.16, 0.0
    yaw = arm_heading(x, y)
    G = np.array([x, y, TABLETOP_HALF_HEIGHT + 0.0045])
    q, ok = solve_grasp_ik(G, np.radians(45))
    q_up, ok2 = solve_grasp_ik(G + [0, 0, 0.06], np.radians(45), q_init=q)
    if not (ok and ok2):
        return False, "IK"
    p = torch.tensor([[x, y, TABLETOP_HALF_HEIGHT]], dtype=torch.float32)
    # Prizma: bir köşesi kola dönük olsun ki parmaklar iki yüze bassın; küp: yüzler parmaklara dik.
    qz = torch.tensor([[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]], dtype=torch.float32)
    obj.set_pose(type(obj.pose).create_from_pq(p=p, q=qz))
    u.agent.robot.set_qpos(torch.tensor([list(q) + [0.0, 0.0]], dtype=torch.float32))
    for _ in range(5):
        env.step(torch.as_tensor(np.r_[q, 0.0], dtype=torch.float32))
    for _ in range(15):
        env.step(torch.as_tensor(np.r_[q, GRIPPER_CLOSED], dtype=torch.float32))
    z0 = obj.pose.p[0, 2].item()
    for s in np.linspace(0, 1, 20):
        env.step(torch.as_tensor(np.r_[(1 - s) * q + s * q_up, GRIPPER_CLOSED], dtype=torch.float32))
    for _ in range(10):
        env.step(torch.as_tensor(np.r_[q_up, GRIPPER_CLOSED], dtype=torch.float32))
    dz = obj.pose.p[0, 2].item() - z0
    held = bool(u.agent.is_grasping(obj)[0])
    g = robot_state(u.agent.robot.get_qpos()[0].numpy())[-1]
    return dz > 0.05 and held, f"Δz={dz * 1000:.1f} mm, temas={held}, kıskaç={g:.2f} rad"


def main():
    env = gym.make("MahutKol-MultiShapeRelational-v1", obs_mode="state_dict", control_mode="pd_joint_pos")
    ok = step_response(env)
    print("Kavrama (x=0.16, y=0, eğim 45°, 6 cm kaldırma):")
    for shape in ("cube", "cylinder", "triangular_prism"):
        r, msg = grasp(env, shape, seed=0)
        print(f"  {shape:<17} {'TUTTU' if r else 'TUTAMADI'}  {msg}")
        ok = ok and r
    print("SONUÇ:", "GEÇTİ" if ok else "KALDI")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
