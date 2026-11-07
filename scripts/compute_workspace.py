#!/usr/bin/env python3
"""
MahutKol erişilebilir masa çalışma alanını FK/IK ile ÖRNEKLEYEREK hesaplar.

Yöntem:
1. Kaba tarama (FK): eklem uzayından rastgele N konfigürasyon -> kavrama noktası
   G = TCP + D * yaklaşma_yönü. (x, y) hücreleri, kavrama eğimi (yaklaşma yönünün
   düşeyle açısı) bandına göre işaretlenir.
2. İnce doğrulama (IK): her (x, y) hücresi ve her eğim için üç yüksekliğin üçü de
   IK ile çözülebiliyor mu?
     - kavrama: G_z = 0.022 (35 mm nesnenin merkezinin biraz üstü)
     - TOP bırakma: G_z + 0.038 (referansın üstüne yerleştirme)
     - taşıma: G_z + 0.065 (diğer nesnelerin üstünden geçiş)
   Ayrıca IK çözümünde parmakların masaya girmediği (en alçak parmak noktası > 0) kontrol edilir.
3. Aday dikdörtgenin her hücresinin en az bir eğimle erişilebilir olduğu doğrulanır.

Kullanım:
    python3 scripts/compute_workspace.py                 # varsayılan aday dikdörtgeni doğrula
    python3 scripts/compute_workspace.py --fk-samples 2000000 --step 0.01
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mahut_kol.robots.kinematics import (  # noqa: E402
    GRASP_DEPTH,
    GRASP_TILTS_DEG,
    get_kinematics,
    solve_grasp_ik,
)

GZ = 0.022
HEIGHTS = (GZ, GZ + 0.038, GZ + 0.065)


def fk_scan(n: int, tilts, step: float, xs, ys):
    K = get_kinematics()
    rng = np.random.default_rng(0)
    grids = {t: np.zeros((len(xs), len(ys)), bool) for t in tilts}
    chunk = 500_000
    per_t = {t: [np.zeros((len(xs), len(ys)), bool) for _ in HEIGHTS] for t in tilts}
    for start in range(0, n, chunk):
        Q = rng.uniform(K.q_min, K.q_max, size=(min(chunk, n - start), 5))
        Q[:, 4] = 0.0
        P, A = K.fk_batch(Q)
        G = P + GRASP_DEPTH * A
        tilt = np.degrees(np.arccos(np.clip(-A[:, 2], -1, 1)))
        for t in tilts:
            mt = np.abs(tilt - t) < 3.0
            for hi, h in enumerate(HEIGHTS):
                m = mt & (np.abs(G[:, 2] - h) < 0.005)
                ix = np.round((G[m, 0] - xs[0]) / step).astype(int)
                iy = np.round((G[m, 1] - ys[0]) / step).astype(int)
                ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
                per_t[t][hi][ix[ok], iy[ok]] = True
    for t in tilts:
        grids[t] = per_t[t][0] & per_t[t][1] & per_t[t][2]
    return grids


def ik_grid(tilts, xs, ys):
    """Her hücre için üç yükseklikte IK; komşu hücrenin çözümüyle ısıtılmış başlangıç + az tohum."""
    from mahut_kol.robots.kinematics import DEFAULT_IK_SEEDS

    seeds = DEFAULT_IK_SEEDS[::4]
    grids = {t: np.zeros((len(xs), len(ys)), bool) for t in tilts}
    for t in tilts:
        for i, x in enumerate(xs):
            warm = None
            for j, y in enumerate(ys):
                q = warm
                ok = True
                for hi, h in enumerate(HEIGHTS):
                    q, o = solve_grasp_ik(np.array([x, y, h]), np.radians(t), q_init=q, seeds=seeds)
                    if not o:
                        ok = False
                        break
                    if hi == 0:
                        warm = q
                grids[t][i, j] = ok
        print(f"   eğim {t}° tamam", flush=True)
    return grids


def print_grid(title, g, xs, ys):
    print(f"{title} (erişilebilir hücre: {int(g.sum())})", flush=True)
    print("       y:" + "".join(f"{y * 100:5.0f}" for y in ys))
    for i, x in enumerate(xs):
        print(f"x={x:5.2f}  " + "".join("    #" if v else "    ." for v in g[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fk-samples", type=int, default=1_000_000)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--rect", type=float, nargs=4, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    ap.add_argument("--skip-ik", action="store_true")
    args = ap.parse_args()

    from mahut_kol.envs.multi_shape_task import (
        WORKSPACE_X_MAX, WORKSPACE_X_MIN, WORKSPACE_Y_MAX, WORKSPACE_Y_MIN,
    )
    rect = args.rect or (WORKSPACE_X_MIN, WORKSPACE_X_MAX, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX)
    xs = np.round(np.arange(0.06, 0.28 + 1e-9, args.step), 4)
    ys = np.round(np.arange(-0.22, 0.18 + 1e-9, args.step), 4)
    tilts = GRASP_TILTS_DEG

    print(f"1) FK taraması: {args.fk_samples} rastgele konfigürasyon, kavrama derinliği D={GRASP_DEPTH} m")
    fk = fk_scan(args.fk_samples, tilts, args.step, xs, ys)
    for t in tilts:
        print_grid(f"   FK eğim {t}°", fk[t], xs, ys)

    if args.skip_ik:
        return
    print("2) IK doğrulaması (üç yükseklik + parmak-masa açıklığı), aday dikdörtgen ± 4 cm")
    x0, x1, y0, y1 = rect
    xs = xs[(xs >= x0 - 0.04 - 1e-9) & (xs <= x1 + 0.04 + 1e-9)]
    ys = ys[(ys >= y0 - 0.04 - 1e-9) & (ys <= y1 + 0.04 + 1e-9)]
    ik = ik_grid(tilts, xs, ys)
    union = np.zeros_like(next(iter(ik.values())))
    for t in tilts:
        print_grid(f"   IK eğim {t}°", ik[t], xs, ys)
        union |= ik[t]
    print_grid("   IK birleşim (herhangi bir eğim)", union, xs, ys)

    mx = (xs >= x0 - 1e-9) & (xs <= x1 + 1e-9)
    my = (ys >= y0 - 1e-9) & (ys <= y1 + 1e-9)
    sub = union[np.ix_(mx, my)]
    print(f"3) Aday dikdörtgen x[{x0}, {x1}] y[{y0}, {y1}]: {int(sub.sum())}/{sub.size} hücre erişilebilir")
    if sub.all():
        print("   ✅ dikdörtgenin tamamı erişilebilir")
    else:
        print("   ❌ erişilemeyen hücreler var")
        sys.exit(1)


if __name__ == "__main__":
    main()
