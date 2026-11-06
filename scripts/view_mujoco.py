#!/usr/bin/env python3
"""
MuJoCo İnteraktif Model ve Sahne Görüntüleyici

Kullanım:
    python3 scripts/view_mujoco.py              # Komple sahneyi açar (Masa, küp, kameralar, robot)
    python3 scripts/view_mujoco.py --robot-only # Sadece robot kolunu açar
    python3 scripts/view_mujoco.py --path assets/mahut_kol_robot.xml
"""

import argparse
import os
import sys
import time

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("❌ 'mujoco' kütüphanesi bulunamadı. Yüklemek için: pip install mujoco")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="MahutKol MuJoCo Model Görüntüleyici")
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Açılacak MJCF (.xml) dosya yolu",
    )
    parser.add_argument(
        "--robot-only",
        action="store_true",
        help="Komple sahne yerine sadece robot kol modelini aç",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.path:
        xml_path = args.path
    elif args.robot_only:
        xml_path = os.path.join(project_root, "assets", "mahut_kol_robot.xml")
    else:
        xml_path = os.path.join(project_root, "assets", "mahut_kol_scene.xml")

    if not os.path.exists(xml_path):
        print(f"❌ Model dosyası bulunamadı: {xml_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"🦾 MuJoCo Modeli Yükleniyor: {xml_path}")
    print("=" * 60)
    print("💡 Kontroller:")
    print("  - Sol Fare Tuşu: Sahneyi döndürme (Rotate)")
    print("  - Sağ Fare Tuşu / Kaydırma: Yakınlaştırma / Uzaklaştırma (Zoom)")
    print("  - Orta Fare Tuşu / Shift + Sol Fare: Sahneyi kaydırma (Pan)")
    print("  - Boşluk (Space): Simülasyonu duraklat / devam ettir")
    print("  - Tab: Kontrol panelini / eklem açılarını aç/kapat")
    print("  - Ctrl + Sol Fare: Nesneleri / robot parçalarını tutup çekme")
    print("  - ESC veya Pencereyi Kapat: Çıkış")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()

            # Zaman adımına uygun bekleme (gerçek zamanlı akış)
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
