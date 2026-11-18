#!/usr/bin/env python3
"""
Yeni kolun Fusion MJCF'inden (Yeni_model/arm_mujoco.xml) ManiSkill URDF'i üretir.

MuJoCo modeli derler, q = 0'da her gövdenin dünya pozunu alır ve gövdeleri hareketli
eklemlere göre URDF link'lerinde toplar. Mesh'ler MuJoCo'nun derlediği (metre, gövde
çerçevesine taşınmış) haliyle STL olarak yazılır, böylece geom pozu doğrudan kullanılır.

Dönüşümler:
- Kol sıfır pozda +x'e baksın diye 180° yaw; j1 ekseni (x, y) = (0, 0), zemin z = 0.
- j1..j5 -> joint0..joint4 (link0..link4). j6 (kıskaç servosu) URDF'te eklem değil: horn6
  link4'e sabitlenir, parmaklar iki prismatik eklem (0..0.018 m), servo açısı kontrolcüde
  parmaklara eşlenir (bkz. robots/mahut_robot.py).
- link4 ve TCP çerçevesi: x = yaklaşma (= joint4 ekseni), y = parmak açılma doğrultusu.
- Mesh'ler çarpışma olarak da eklenir (SAPIEN dışbükey zarf alır); MJCF'te kol mesh'lerinin
  çarpışması kapalıydı.

Kullanım: python3 scripts/mjcf_to_urdf.py [--mjcf Yeni_model/arm_mujoco.xml]
"""

import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
ARM = {"j1": 0, "j2": 1, "j3": 2, "j4": 3, "j5": 4}
FINGERS = {"finger_left": "finger_left", "finger_right": "finger_right"}


def rpy(R):
    """R = Rz(y) Ry(p) Rx(r) -> (r, p, y)."""
    p = np.arcsin(-np.clip(R[2, 0], -1, 1))
    if abs(np.cos(p)) > 1e-9:
        return np.arctan2(R[2, 1], R[2, 2]), p, np.arctan2(R[1, 0], R[0, 0])
    return np.arctan2(-R[1, 2], R[1, 1]), p, 0.0


def origin(p, R):
    f = lambda v: " ".join(f"{x:.8g}" for x in v)
    return f'<origin xyz="{f(p)}" rpy="{f(rpy(R))}"/>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjcf", default=str(ROOT / "Yeni_model/arm_mujoco.xml"))
    ap.add_argument("--out", default=str(ROOT / "assets/urdf/mahut_kol.urdf"))
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.mjcf)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    # Dünya düzeltmesi: x' = W (x - c)
    j1 = m.joint("j1").id
    c = np.array([d.xanchor[j1][0], d.xanchor[j1][1], 0.0])
    W = np.diag([-1.0, -1.0, 1.0])
    P = lambda x: W @ (np.asarray(x) - c)
    Rw = lambda R: W @ np.asarray(R).reshape(3, 3)

    # Link çerçeveleri (q = 0)
    a4 = W @ d.xaxis[m.joint("j5").id]
    g = m.body("gripper").id
    f4 = Rw(d.xmat[g])[:, 0]
    f4 = f4 - (f4 @ a4) * a4
    f4 /= np.linalg.norm(f4)
    R4 = np.stack([a4, f4, np.cross(a4, f4)], axis=1)
    frames = {"base_link": (np.zeros(3), np.eye(3))}
    for jn, i in ARM.items():
        frames[f"link{i}"] = (P(d.xanchor[m.joint(jn).id]), R4 if i == 4 else np.eye(3))
    for b in FINGERS:
        frames[f"{b}_link"] = (P(d.xpos[m.body(b).id]), R4)

    def owner(b):
        while b > 0:
            name = m.body(b).name
            if name in FINGERS:
                return f"{name}_link"
            if name == "gripper":
                return "link4"
            for k in range(m.body_jntnum[b]):
                jn = m.joint(m.body_jntadr[b] + k).name
                if jn in ARM:
                    return f"link{ARM[jn]}"
                if jn == "j6":
                    return "link4"
            b = m.body_parentid[b]
        return "base_link"

    bodies = {}
    for b in range(1, m.nbody):
        if m.body(b).name == "cube":
            continue
        bodies.setdefault(owner(b), []).append(b)

    mesh_dir = Path(args.out).parent.parent / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    colors = {}
    links_xml = []
    for link, (o, RL) in frames.items():
        parts = []
        # Kütle / atalet (bileşik, link çerçevesinde)
        ms = np.array([m.body_mass[b] for b in bodies[link]])
        cs = np.array([P(d.xipos[b]) for b in bodies[link]])
        M = ms.sum()
        C = (ms[:, None] * cs).sum(0) / M
        I = np.zeros((3, 3))
        for b, mb, cb in zip(bodies[link], ms, cs):
            Ri = Rw(d.ximat[b])
            r = cb - C
            I += Ri @ np.diag(m.body_inertia[b]) @ Ri.T + mb * (r @ r * np.eye(3) - np.outer(r, r))
        I = RL.T @ I @ RL
        parts.append(
            f'    <inertial>{origin(RL.T @ (C - o), np.eye(3))}<mass value="{M:.8g}"/>'
            f'<inertia ixx="{I[0,0]:.8g}" ixy="{I[0,1]:.8g}" ixz="{I[0,2]:.8g}" '
            f'iyy="{I[1,1]:.8g}" iyz="{I[1,2]:.8g}" izz="{I[2,2]:.8g}"/></inertial>'
        )
        for gi in range(m.ngeom):
            if m.geom_bodyid[gi] not in bodies[link]:
                continue
            Rg = RL.T @ Rw(d.geom_xmat[gi])
            pg = RL.T @ (P(d.geom_xpos[gi]) - o)
            rgba = tuple(np.round(m.geom_rgba[gi], 3))
            mat = colors.setdefault(rgba, f"mat{len(colors)}")
            if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gi]
                v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                fc = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                h = hashlib.sha1(v.tobytes() + fc.tobytes()).hexdigest()
                if h not in written:  # aynı servo/rulman mesh'leri tek dosya
                    name = m.mesh(mid).name
                    trimesh.Trimesh(v.copy(), fc.copy(), process=False).export(mesh_dir / f"{name}.stl")
                    written[h] = name
                geo = f'<mesh filename="../meshes/{written[h]}.stl"/>'
            elif m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX:
                geo = '<box size="{} {} {}"/>'.format(*(2 * m.geom_size[gi]))
            else:
                raise ValueError(f"Desteklenmeyen geom tipi: {m.geom_type[gi]}")
            parts.append(f'    <visual>{origin(pg, Rg)}<geometry>{geo}</geometry><material name="{mat}"/></visual>')
            parts.append(f"    <collision>{origin(pg, Rg)}<geometry>{geo}</geometry></collision>")
        links_xml.append(f'  <link name="{link}">\n' + "\n".join(parts) + "\n  </link>")

    def joint(name, jtype, parent, child, axis=None, lo=None, hi=None, effort=3.4, vel=3.0):
        po, pR = frames[parent]
        co, cR = frames[child]
        s = f'  <joint name="{name}" type="{jtype}"><parent link="{parent}"/><child link="{child}"/>'
        s += origin(pR.T @ (co - po), pR.T @ cR)
        if axis is not None:
            ax = cR.T @ axis
            s += '<axis xyz="{:.8g} {:.8g} {:.8g}"/>'.format(*ax)
            s += f'<limit lower="{lo:.6g}" upper="{hi:.6g}" effort="{effort}" velocity="{vel}"/>'
        return s + "</joint>"

    joints_xml = []
    parent = "base_link"
    for jn, i in ARM.items():
        jid = m.joint(jn).id
        lo, hi = m.jnt_range[jid]
        joints_xml.append(joint(f"joint{i}", "revolute", parent, f"link{i}", W @ d.xaxis[jid], lo, hi))
        parent = f"link{i}"
    gR = Rw(d.xmat[g])
    for b in FINGERS:
        jid = m.body_jntadr[m.body(b).id]
        lo, hi = m.jnt_range[jid]
        joints_xml.append(
            joint(b, "prismatic", "link4", f"{b}_link", gR @ m.jnt_axis[jid], lo, hi, effort=20.0, vel=0.2)
        )
    tcp = P(d.site_xpos[m.site("tcp").id])
    frames["link_gripper_center"] = (tcp, R4)
    links_xml.append('  <link name="link_gripper_center"/>')
    joints_xml.append(joint("joint_gripper_center", "fixed", "link4", "link_gripper_center"))

    mats = "\n".join(
        f'  <material name="{n}"><color rgba="{" ".join(map(str, c))}"/></material>' for c, n in colors.items()
    )
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!-- Üretildi: scripts/mjcf_to_urdf.py (Yeni_model/arm_mujoco.xml). Elle düzenleme. -->\n"
        f'<robot name="mahut_kol">\n{mats}\n' + "\n".join(links_xml) + "\n" + "\n".join(joints_xml) + "\n</robot>\n"
    )
    Path(args.out).write_text(xml)
    print(f"{args.out}: {len(frames)} link, {len(joints_xml)} eklem, {len(written)} mesh -> {mesh_dir}")
    print(f"TCP (q=0): {np.round(tcp, 4)}, yaklaşma: {np.round(a4, 3)}, toplam kütle: "
          f"{sum(m.body_mass[b] for v in bodies.values() for b in v):.3f} kg")


if __name__ == "__main__":
    main()
