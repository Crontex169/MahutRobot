#!/usr/bin/env python3
"""
ManiSkill 3 - MultiShapeRelationalTask Canlı Simülasyon Görselleştiricisi.

Bu betik:
1. 9 Şekil-Renk nesnesi ve SO-100 robot kolunu içeren ortamı ('MahutKol-MultiShapeRelational-v1') SAPIEN Viewer ile açar.
2. Sahneyi fare ile 360 derece döndürmenizi ve yakınlaştırmanızı sağlar.
3. İki farklı mod sunar:
   - Varsayılan Mod: Epizot modunda aktif nesneler masada, pasifler yer altında; bağıl hedef pozisyonu gösterilir.
   - Tümünü Göster Modu (`--show-all-9`): 9 nesnenin tamamını (3x3 grid: Küp, Silindir, Prizma / Kırmızı, Yeşil, Mavi) masaya dizer.

Kullanım:
    python3 scripts/view_multi_shape.py                   # İnteraktif 3D Simülasyon Penceresi
    python3 scripts/view_multi_shape.py --show-all-9       # 9 nesnenin tamamını masada 3x3 dizilmiş gösterir
    python3 scripts/view_multi_shape.py --save-snapshot    # Ekran görüntüsü alır (assets/multi_shape_view.png)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# src dizinini Python yoluna ekle
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.utils.structs import Pose
import mahut_kol.envs
from mahut_kol.envs.multi_shape_task import MultiShapeRelationalTask, RELATIONS
from mahut_kol.envs.shape_objects import OBJECT_COLORS, OBJECT_SHAPES, enable_actor_collision


def main():
    parser = argparse.ArgumentParser(description="MahutKol Multi-Shape Relational Task Görselleştirici")
    parser.add_argument(
        "--show-all-9",
        action="store_true",
        help="9 nesnenin tamamını masaya 3x3 ızgara şeklinde dizerek gösterir.",
    )
    parser.add_argument(
        "--save-snapshot",
        action="store_true",
        help="Görüntüyü 'assets/multi_shape_view.png' olarak kaydeder ve çıkar.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Görselleştirici açıkken otomatik geçiş yapılacak epizot sayısı (varsayılan: 10)",
    )
    args = parser.parse_args()

    env_id = "MahutKol-MultiShapeRelational-v1"

    if args.save_snapshot:
        import cv2
        print(f"📷 {env_id} sahnesinin ekran görüntüsü alınıyor...")
        env = gym.make(env_id, render_mode="rgb_array")
        obs, _ = env.reset()

        if args.show_all_9:
            # 9 nesneyi 3x3 grid olarak masaya diz
            unwrapped: MultiShapeRelationalTask = env.unwrapped
            shapes = ["cube", "cylinder", "triangular_prism"]
            colors = ["red", "green", "blue"]
            for row, color in enumerate(colors):
                for col, shape in enumerate(shapes):
                    key = f"{color}_{shape}"
                    x = 0.13 + col * 0.045
                    y = -0.06 + row * 0.06
                    z = 0.0175
                    enable_actor_collision(unwrapped.manipulation_objects[key])
                    pose = Pose.create_from_pq(p=torch.tensor([[x, y, z]], device=env.unwrapped.device))
                    unwrapped.manipulation_objects[key].set_pose(pose)

        img = env.render()
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 4:
            img = img[0]

        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out_path = _REPO_ROOT / "assets" / "multi_shape_view.png"
        cv2.imwrite(str(out_path), bgr)
        env.close()
        print(f"✅ Görüntü kaydedildi: {out_path}")
        return

    print("=" * 70)
    print(f"🤖 SAPIEN 3D Simülasyon Görselleştirici Başlatılıyor: {env_id}")
    print("=" * 70)
    print("💡 Fare Kontrolleri:")
    print("  - Sol Tık + Sürükle : Sahneyi 360° Döndür (Orbit)")
    print("  - Sağ Tık / Tekerlek: Yakınlaştır / Uzaklaştır (Zoom)")
    print("  - Orta Tık          : Sahneyi Kaydır (Pan)")
    print("💡 Klavye / Kontrol:")
    print("  - Her 3 saniyede bir sahne yeni nesne kombinasyonuyla yenilenir.")
    print("  - Çıkış için pencereyi kapatın veya terminalde Ctrl+C yapın.")
    print("=" * 70)

    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode="pd_joint_target_delta_pos",
        render_mode="human",
    )

    unwrapped: MultiShapeRelationalTask = env.unwrapped

    try:
        if args.show_all_9:
            print("\n>>> MOD: 9 NESNENİN TAMAMI MASADA GÖSTERİLİYOR (3x3 Grid) <<<")
            env.reset()
            shapes = ["cube", "cylinder", "triangular_prism"]
            colors = ["red", "green", "blue"]
            for row, color in enumerate(colors):
                for col, shape in enumerate(shapes):
                    key = f"{color}_{shape}"
                    x = 0.13 + col * 0.048
                    y = -0.06 + row * 0.06
                    z = 0.0175
                    enable_actor_collision(unwrapped.manipulation_objects[key])
                    pose = Pose.create_from_pq(p=torch.tensor([[x, y, z]], device=unwrapped.device))
                    unwrapped.manipulation_objects[key].set_pose(pose)
                    print(f"  [+] {key:25s} @ x={x:.3f}, y={y:.3f}, z={z:.3f}")

            viewer = env.render()
            while True:
                if hasattr(viewer, "closed") and viewer.closed:
                    break
                # Nesneleri sabit tut, robotu hazır duruşta göster
                env.step(torch.zeros(env.action_space.shape, device=unwrapped.device))
                viewer = env.render()
                time.sleep(0.02)
        else:
            print("\n>>> MOD: DİNAMİK BAĞIL YERLEŞİM VE NESNE HAVUZU MODU <<<")
            for ep in range(1, args.episodes + 1):
                obs, _ = env.reset()
                manip_id = unwrapped.manipulated_object_id
                ref_id = unwrapped.reference_object_id
                distractors = unwrapped.distractor_object_ids
                relation = unwrapped.current_relation

                target_xyz = unwrapped.get_relational_target_position().squeeze().tolist()

                print(f"\n[Epizot {ep}/{args.episodes}]")
                print(f"  🎯 Taşınacak Nesne : {manip_id}")
                print(f"  📍 Referans Nesne  : {ref_id}")
                print(f"  🎲 Çeldiriciler    : {distractors}")
                print(f"  📐 Bağıl Hedef     : {relation} -> [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")
                print("  💤 Kullanılmayan nesneler yer altında (z = -10.0 m) gizlendi.")

                viewer = env.render()
                # 3 saniye boyunca bu sahneyi göster ve hafif robot nefes alma hareketi yap
                start_t = time.time()
                while time.time() - start_t < 3.5:
                    if hasattr(viewer, "closed") and viewer.closed:
                        return
                    action = torch.zeros(env.action_space.shape, device=unwrapped.device)
                    env.step(action)
                    viewer = env.render()
                    time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nÇıkış yapılıyor...")
    finally:
        env.close()
        print("Simülasyon kapatıldı.")


if __name__ == "__main__":
    main()
