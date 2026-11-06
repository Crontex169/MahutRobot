#!/usr/bin/env python3
"""
SO-100 (SO-ARM100) Robot Kolu MuJoCo Görselleştirici

Bu betik, The Robot Studio / Hugging Face LeRobot tarafından geliştirilen
açık kaynaklı SO-100 (SO-ARM100) robot kolunu MuJoCo simülasyonunda görselleştirir.

Kullanım:
    python3 view_so100.py
    python3 view_so100.py --model-only
    python3 view_so100.py --keyframe rest
    python3 view_so100.py --passive
"""

import argparse
import os
import sys
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("HATA: 'mujoco' kütüphanesi bulunamadı!")
    print("Yüklemek için: pip install mujoco")
    sys.exit(1)


def get_model_path(use_scene: bool = True) -> Path:
    """Model XML dosyasının yolunu belirler."""
    current_dir = Path(__file__).resolve().parent
    model_dir = current_dir / "so_arm100"

    if use_scene:
        xml_file = model_dir / "scene.xml"
    else:
        xml_file = model_dir / "so_arm100.xml"

    if not xml_file.exists():
        # Alternatif olarak doğrudan so_arm100.xml kontrol et
        fallback = model_dir / "so_arm100.xml"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"Model dosyası bulunamadı: {xml_file}")

    return xml_file


def print_robot_info(model: mujoco.MjModel, data: mujoco.MjData):
    """Robot ve eklem bilgilerini ekrana yazdırır."""
    print("=" * 60)
    print("SO-100 (SO-ARM100) Robot Kolu - MuJoCo Simülasyonu")
    print("=" * 60)
    print(f"Toplam Eklem Sayısı (nq): {model.nq}")
    print(f"Serbestlik Derecesi (nv): {model.nv}")
    print(f"Aktüatör Sayısı (nu):     {model.nu}")
    print("-" * 60)
    print("Eklemler:")
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        print(f"  [{i}] {joint_name} (Tip: {joint_type})")
    print("-" * 60)
    if model.nkey > 0:
        print("Mevcut Pozlar (Keyframes):")
        for i in range(model.nkey):
            key_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) or f"key_{i}"
            print(f"  [{i}] {key_name}")
    print("=" * 60)
    print("Kamera ve Etkileşim Kontrolleri:")
    print("  - Sol Tık + Sürükle      : Kamerayı döndür")
    print("  - Sağ Tık + Sürükle     : Yakınlaştır / Uzaklaştır")
    print("  - Orta Tık + Sürükle    : Kamerayı kaydır (Pan)")
    print("  - Çift Tık              : Seçilen gövdeye odaklan")
    print("  - Ctrl + Sağ Tık Sürükle: Robota kuvvet uygula / çek")
    print("  - Boşluk (Space)        : Simülasyonu duraklat / devam ettir")
    print("  - Backspace             : Simülasyonu sıfırla")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="SO-100 MuJoCo Görselleştirici")
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Zemin ve sahne ışıkları olmadan sadece robot modelini yükler",
    )
    parser.add_argument(
        "--keyframe",
        type=str,
        default="home",
        choices=["home", "rest", "none"],
        help="Başlangıç pozisyonu (home, rest veya none)",
    )
    parser.add_argument(
        "--passive",
        action="store_true",
        help="Pasif görselleştirici modunda adım adım çalıştırır",
    )
    args = parser.parse_args()

    xml_path = get_model_path(use_scene=not args.model_only)
    print(f"Model yükleniyor: {xml_path}")

    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"HATA: Model yüklenirken bir sorun oluştu:\n{e}")
        sys.exit(1)

    # Başlangıç pozunu ayarla
    if args.keyframe != "none":
        key_id = -1
        for i in range(model.nkey):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
            if name == args.keyframe:
                key_id = i
                break
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key_id)
            print(f"Robot '{args.keyframe}' pozuna ayarlandı.")
        else:
            mujoco.mj_resetData(model, data)
    else:
        mujoco.mj_resetData(model, data)

    # Eklem ve simülasyon bilgilerini yazdır
    print_robot_info(model, data)

    print("\nSimülasyon penceresi açılıyor. Kapatmak için pencereyi kapatın veya Ctrl+C tuşlayın...")

    try:
        if args.passive:
            # launch_passive modu: döngü içinde adım adım simülasyon
            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running():
                    step_start = time.time()
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time_until_next_step = model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            # Tam donanımlı interaktif MuJoCo Simulate GUI
            mujoco.viewer.launch(model, data)
    except KeyboardInterrupt:
        print("\nKullanıcı tarafından durduruldu.")
    finally:
        print("Görselleştirici kapatıldı.")
        os._exit(0)


if __name__ == "__main__":
    main()
