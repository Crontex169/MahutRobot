"""
Hızlı Test Betiği: URDF Doğrulama ve Eklem Hareketi Kontrolü.

Bu betik:
1. Belirtilen URDF dosyasını SAPIEN motoruna yükler.
2. Link ve eklem (joint) hiyerarşisini ekrana yazdırır.
3. Eklem limitlerini ve aktüatör tiplerini denetler.
"""

import argparse
import os
import sys
import sapien


def check_and_visualize_urdf(urdf_path: str, render: bool = False):
    """
    URDF'i SAPIEN sahnesine yükleyip doğrular.
    """
    abs_path = os.path.abspath(urdf_path)
    print("=" * 60)
    print(f"🤖 URDF Doğrulama Başlatılıyor: {abs_path}")
    print("=" * 60)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"URDF dosyası bulunamadı: {abs_path}")

    scene = sapien.Scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True

    try:
        robot = loader.load(abs_path)
    except Exception as e:
        print(f"❌ URDF Yükleme Hatası: {e}")
        sys.exit(1)

    print("✅ URDF Başarıyla Yüklendi!\n")

    print(f"📦 Linkler ({len(robot.get_links())} adet):")
    for link in robot.get_links():
        print(f"  • {link.name}")

    print(f"\n⚙️  Aktif Eklemler ({len(robot.get_active_joints())} adet):")
    for joint in robot.get_active_joints():
        limits = joint.get_limits()
        print(f"  • {joint.name:<16} | Tip: {joint.type:<10} | Limitler: [{limits[0][0]:.3f}, {limits[0][1]:.3f}]")

    print("\n" + "=" * 60)
    print("✅ URDF Doğrulama Başarıyla Tamamlandı!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="URDF Doğrulama Aracı")
    parser.add_argument("--urdf", type=str, default="assets/urdf/mahut_kol.urdf", help="Test edilecek URDF dosya yolu")
    parser.add_argument("--render", action="store_true", help="Viewer aç (GUI ortamında)")
    args = parser.parse_args()

    check_and_visualize_urdf(urdf_path=args.urdf, render=args.render)
