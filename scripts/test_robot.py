#!/usr/bin/env python3
"""
Otomatik Hareket Testi: Robotun tüm aktüatörlerini alt limite, üst limite ve merkeze sürer.

Kullanım:
    python3 scripts/test_robot.py             # İnteraktif MuJoCo viewer penceresinde izle
    python3 scripts/test_robot.py --headless  # Penceresiz, terminal raporu ile test et
"""
import os
import sys
import time
import argparse
import numpy as np
import mujoco
try:
    import mujoco.viewer
except ImportError:
    pass

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_XML = os.path.join(_PROJECT_ROOT, "assets", "mahut_kol_robot.xml")


def run(m, d, viewer=None):
    MOVE = 2.0  # seconds per target
    for i in range(m.nu):
        lo, hi = m.actuator_ctrlrange[i]
        qadr = m.jnt_qposadr[m.actuator_trnid[i, 0]]
        for target in (lo, hi, np.clip(0, lo, hi)):
            start = d.ctrl[i]
            steps = int(MOVE / m.opt.timestep)
            for k in range(steps):
                d.ctrl[i] = start + (target - start) * min(1, 2 * k / steps)
                mujoco.mj_step(m, d)
                if viewer and k % 10 == 0:
                    if not viewer.is_running():
                        return
                    viewer.sync()
                    time.sleep(10 * m.opt.timestep)
            assert np.all(np.isfinite(d.qpos)), "simülasyon patladı (patlama hatası)"
            err = abs(d.qpos[qadr] - target)
            flag = "  <-- ulaşılamadı (temas veya zayıf kp)" if err > 0.05 * (hi - lo) else ""
            print(f"{m.actuator(i).name:8s} hedef {target:+.3f}  ulaşılan {d.qpos[qadr]:+.3f}{flag}")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo Robot Eklem Limit ve Hareket Testi")
    parser.add_argument("--model", type=str, default=_DEFAULT_XML, help="Model XML yolu")
    parser.add_argument("--headless", action="store_true", help="Penceresiz çalıştır")
    args = parser.parse_args()

    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)

    if args.headless:
        run(m, d)
    else:
        with mujoco.viewer.launch_passive(m, d) as v:
            run(m, d, v)
            while v.is_running():
                v.sync()
                time.sleep(0.02)


if __name__ == "__main__":
    main()
