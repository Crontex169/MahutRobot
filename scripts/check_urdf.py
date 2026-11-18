"""
URDF doğrulama (G1 kapısı): SAPIEN'e yükler, link kütle/atalet ve çarpışma şekillerini,
eklem limitlerini yazdırır; verilen duruşta (varsayılan: ROBOT_REST_QPOS ve q = 0) bitişik
olmayan link'ler arasında kendi kendine çarpışma arar. Sorun varsa çıkış kodu 1.

Kullanım: python3 scripts/check_urdf.py [--urdf assets/urdf/mahut_kol.urdf] [--qpos q0 .. q6]
"""

import argparse
import os
import sys

import numpy as np
import sapien


def self_collisions(robot, scene, qpos):
    """Bitişik olmayan link çiftleri arasındaki penetrasyonlar [(a, b, derinlik_m)]."""
    robot.set_qpos(np.asarray(qpos, dtype=np.float32))
    for lk in robot.get_links():
        for s in lk.get_collision_shapes():
            s.set_collision_groups([1, 1, 0, 0])  # robot içi çarpışmayı aç
    scene.step()
    parent = {j.get_child_link().name: j.get_parent_link().name for j in robot.get_joints() if j.get_parent_link()}
    out = []
    for c in scene.get_contacts():
        a, b = c.bodies[0].entity.name, c.bodies[1].entity.name
        if parent.get(a) == b or parent.get(b) == a:
            continue
        depth = min((p.separation for p in c.points), default=0.0)
        if depth < -1e-4:
            out.append((a, b, depth))
    return out


def main():
    ap = argparse.ArgumentParser(description="URDF Doğrulama Aracı")
    ap.add_argument("--urdf", default="assets/urdf/mahut_kol.urdf")
    ap.add_argument("--qpos", type=float, nargs="*", help="Kontrol edilecek duruş (7 değer)")
    args = ap.parse_args()

    scene = sapien.Scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(os.path.abspath(args.urdf))
    ok = True

    print(f"Linkler ({len(robot.get_links())}):  kütle kg | atalet (asal) kg·m² | çarpışma şekli")
    total = 0.0
    for lk in robot.get_links():
        n = len(lk.get_collision_shapes())
        total += lk.mass
        print(f"  {lk.name:<20} {lk.mass:7.4f} | {np.array2string(np.asarray(lk.inertia), precision=7)} | {n}")
        if lk.name != "link_gripper_center" and (n == 0 or lk.mass <= 0 or min(lk.inertia) <= 0):
            print("    ! çarpışma şekli/kütle/atalet eksik")
            ok = False
    print(f"  toplam {total:.3f} kg")

    print(f"\nAktif eklemler ({len(robot.get_active_joints())}):")
    for j in robot.get_active_joints():
        lo, hi = j.get_limits()[0]
        print(f"  {j.name:<14} {j.type:<10} [{lo:.4f}, {hi:.4f}]")

    poses = [args.qpos] if args.qpos else []
    if not args.qpos:
        from mahut_kol.envs.multi_shape_task import ROBOT_REST_QPOS

        poses = [ROBOT_REST_QPOS, [0.0] * len(robot.get_active_joints())]
    for q in poses:
        hits = self_collisions(robot, scene, q)
        print(f"\nKendi kendine çarpışma @ {np.round(q, 3).tolist()}: {'yok' if not hits else ''}")
        for a, b, dep in hits:
            print(f"  {a} <-> {b}: {dep * 1000:.1f} mm")
        ok = ok and not hits

    print("\nSONUÇ:", "GEÇTİ" if ok else "KALDI")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
