#!/usr/bin/env python3
"""
Taşıma Kapasitesi (Payload) ve Statik Tork Analiz Aracı.

Kullanım:
    python3 scripts/load_test.py
"""
import itertools
import os
import numpy as np
import mujoco

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_XML = os.path.join(_PROJECT_ROOT, "assets", "mahut_kol_robot.xml")

PAYLOADS_G = range(0, 1001, 100)
SERVO_TORQUE = 35 * 0.0980665  # stall, 35 kg*cm -> N*m
RATED_TORQUE = 11.8 * 0.0980665  # continuous rated, 11.8 kg*cm
HOLD_KP = 200

spec = mujoco.MjSpec.from_file(_DEFAULT_XML)
m0 = spec.compile()
d0 = mujoco.MjData(m0)
mujoco.mj_forward(m0, d0)

# Grasp point: midway between the two clamp geoms, expressed in link4's frame.
clamps = [
    g for g in range(m0.ngeom)
    if m0.geom_bodyid[g] in (m0.body("finger_left").id, m0.body("finger_right").id) and m0.geom_contype[g]
]
grasp = np.mean([d0.geom_xpos[g] for g in clamps], axis=0)
L4 = m0.body("link4").id
local = d0.xmat[L4].reshape(3, 3).T @ (grasp - d0.xpos[L4])
payload = spec.body("link4").add_body(name="payload", pos=local)
payload.add_geom(
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.015, 0, 0],
    mass=1e-6,
    contype=0,
    conaffinity=0,
    rgba=[0.9, 0.2, 0.2, 1],
)
m = spec.compile()
d = mujoco.MjData(m)
P = m.body("payload").id
m.geom_contype[:] = 0
m.geom_conaffinity[:] = 0
act = [m.actuator(f"joint{i}").id for i in range(1, 5)]  # joint0 (yaw) carries no gravity load
m.actuator_gainprm[act, 0] = HOLD_KP
m.actuator_biasprm[act, 1] = -HOLD_KP

arm = [m.joint(f"joint{i}") for i in range(1, 5)]
grids = [np.arange(j.range[0], j.range[1] + 1e-9, np.radians(10)) for j in arm[:3]] + [np.radians([-90, 0, 90])]
tau = np.zeros(m.nv)


def set_payload(kg):
    m.body_mass[P] = max(kg, 1e-6)
    m.body_inertia[P] = 0.4 * m.body_mass[P] * 0.015 ** 2


def worst_case():
    worst, pose = np.zeros(4), [None] * 4
    for q in itertools.product(*grids):
        d.qpos[:] = 0
        d.qvel[:] = 0
        for j, v in zip(arm, q):
            d.qpos[j.qposadr[0]] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_rne(m, d, 0, tau)
        t = np.abs([tau[j.dofadr[0]] for j in arm])
        for i in np.flatnonzero(t > worst):
            worst[i], pose[i] = t[i], q
    return worst, pose


def holds(q):
    mujoco.mj_resetData(m, d)
    for i, (j, v) in enumerate(zip(arm, q)):
        d.qpos[j.qposadr[0]] = v
        d.ctrl[act[i]] = v
    for _ in range(int(3 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return max(abs(d.qpos[j.qposadr[0]] - v) for j, v in zip(arm, q))


def main():
    print(f"Stall {SERVO_TORQUE:.2f} Nm, Rated {RATED_TORQUE:.2f} Nm. % = payı, * sürekli limiti aşıyor, ! stall limiti aşıyor\n")
    print("Yük     joint1          joint2          joint3          joint4      En kötü pozda (joint1)")
    for g in PAYLOADS_G:
        set_payload(g / 1000)
        worst, pose = worst_case()
        err = np.degrees(holds(pose[0]))
        cells = "  ".join(
            f"{w:4.2f}Nm {100 * w / RATED_TORQUE:3.0f}%" + ("!" if w > SERVO_TORQUE else "*" if w > RATED_TORQUE else " ")
            for w in worst
        )
        print(f"{g:4d} g  {cells}  " + (f"TUTUYOR (sapma {err:.1f} deg)" if err < 5 else f"DÜŞÜYOR (sapma {err:.0f} deg)"))

    r, mu = 0.0072, 0.3
    for g in (100, 300, 500):
        F = g / 1000 * 9.81 / (2 * mu)
        print(f"\nGripper {g} g: Parmak kuvveti {F:.1f} N, Servo torku {2 * F * r:.2f} Nm ({100 * 2 * F * r / RATED_TORQUE:.0f}% rated)", end="")
    print()


if __name__ == "__main__":
    main()
