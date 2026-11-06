"""
İnsan Gösterimi (Demonstration) Kaydedici Şablonu.

Bu betik:
1. ManiSkill ortamını ("MahutKol-PickCube-v1") görsel render modunda açar.
2. TeleopController üzerinden gelen aksiyonları ortama iletir (`env.step`).
3. Her adımda:
   - Gözlemleri (Kamera RGB/Depth görüntüleri, robot qpos/qvel)
   - Uygulanan eylemleri (Actions)
   - Ödül ve başarı durumunu
   bir HDF5 veya trajectory dosyasına kaydeder.
4. Başarısız olan bölümleri filtreleyip yalnızca başarılı olanları kaydetmenize olanak tanır.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional
import h5py
import numpy as np

# Proje ve src/ dizinlerini Python yoluna ekle
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "../src"))
_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from Teleoperasyonlar.teleop_agent import TeleopController
except ImportError:
    from mahut_kol.teleop.teleop_agent import TeleopController


def record_demonstrations(
    env_id: str,
    output_path: str,
    num_episodes: int = 50,
    device_type: str = "gamepad",
    control_freq: int = 20,
):
    """
    Teleoperasyon döngüsünü çalıştırıp gösterimleri HDF5 formatında kaydeder.
    """
    # 1. Ortamın başlatılması
    try:
        import gymnasium as gym
        if env_id.startswith("MahutKol"):
            try:
                import mahut_kol.envs
            except Exception as e:
                print(f"[UYARI] MahutKol ortamı yüklenirken hata oluştu: {e}")
                print("[BİLGİ] Robot URDF dosyası henüz hazır değilse, önce standart ManiSkill ortamını test edebilirsiniz:")
                print("       python src/mahut_kol/teleop/record_demos.py --env-id PickCube-v1 --device ps4\n")
                raise e

        # ManiSkill ortamını başlat (End-Effector delta pose kontrol moduyla)
        print(f"[BİLGİ] '{env_id}' simülasyon ortamı başlatılıyor...")
        env = gym.make(
            env_id,
            obs_mode="rgbd",
            render_mode="human",
            control_mode="pd_ee_delta_pose",
        )
    except ModuleNotFoundError as e:
        print(f"[HATA] Gerekli kütüphane bulunamadı: {e}")
        print("Lütfen sanal ortamınızın aktif olduğundan ve bağımlılıkların yüklendiğinden emin olun:")
        print("  pip install -r requirements.txt")
        return
    except Exception as e:
        print(f"[HATA] Ortam açılamadı: {e}")
        return

    # 2. Teleoperasyon Kontrolcüsünün Başlatılması
    controller = TeleopController(device_type=device_type)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print("=" * 60)
    print(f"🎬 GÖSTERİM KAYDI BAŞLATILDI: {output_path}")
    print(f"🎮 Kontrol Cihazı : {device_type.upper()}")
    print(f"🎯 Hedef Epizot   : {num_episodes}")
    print(f"🔄 Kontrol Frekansı: {control_freq} Hz")
    print("=" * 60)
    print("İpuçları:")
    print(" - Başarısız olursanız: PS4'te ÜÇGEN / OPTIONS tuşuna (veya klavyede 'R') basarak epizotu sıfırlayın.")
    print(" - Başarılı olunca    : PS4'te X (CROSS) tuşuna basarak epizotu kaydedin.")
    print(" - Tamamen çıkmak için: Terminalde Ctrl+C yapın.\n")

    h5_file = h5py.File(output_path, "w")
    saved_episodes = 0
    dt = 1.0 / control_freq

    try:
        while saved_episodes < num_episodes:
            print(f"\n▶ Epizot {saved_episodes + 1}/{num_episodes} Başlatılıyor...")
            obs, info = env.reset()

            ep_actions: List[np.ndarray] = []
            ep_states: List[np.ndarray] = []
            ep_rewards: List[float] = []

            step_count = 0
            is_success = False
            user_reset = False

            while True:
                step_start = time.time()

                # Giriş cihazından aksiyon oku
                raw_action = controller.get_action()

                # Aksiyon boyutunu ortamın eylem uzayına uyarla
                act_dim = env.action_space.shape[0]
                if len(raw_action) >= act_dim:
                    action = raw_action[:act_dim]
                else:
                    action = np.pad(raw_action, (0, act_dim - len(raw_action)))

                # Simülasyon adımı
                next_obs, reward, terminated, truncated, info = env.step(action)
                env.render()

                # Bölüm durumunu tampona ekle
                ep_actions.append(action)
                ep_rewards.append(float(reward))
                if isinstance(obs, dict) and "state" in obs:
                    ep_states.append(np.array(obs["state"]))
                elif isinstance(obs, np.ndarray):
                    ep_states.append(obs)

                obs = next_obs
                step_count += 1

                # Başarı ve iptal kontrolleri
                if info.get("success", False) or controller.is_save_requested():
                    is_success = True
                    print(f"  ✓ Görev Başarılı! (Adım: {step_count})")
                    break

                if controller.is_reset_requested():
                    user_reset = True
                    print("  ✗ Epizot kullanıcı tarafından iptal edildi/sıfırlandı.")
                    break

                if terminated or truncated:
                    print("  - Epizot sınırına ulaşıldı.")
                    break

                # FPS senkronizasyonu
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            # Başarılı bölümü HDF5'e yaz
            if is_success and not user_reset and len(ep_actions) > 5:
                grp = h5_file.create_group(f"episode_{saved_episodes}")
                grp.create_dataset("actions", data=np.array(ep_actions, dtype=np.float32))
                grp.create_dataset("rewards", data=np.array(ep_rewards, dtype=np.float32))
                if len(ep_states) > 0:
                    grp.create_dataset("states", data=np.array(ep_states, dtype=np.float32))
                grp.attrs["num_steps"] = len(ep_actions)
                grp.attrs["success"] = True

                saved_episodes += 1
                h5_file.attrs["total_episodes"] = saved_episodes
                h5_file.flush()
                print(f"💾 Epizot {saved_episodes} başarıyla HDF5 dosyasına kaydedildi!")
            else:
                print("  ⟳ Bu epizot kaydedilmedi, yeniden deneniyor.")

    except KeyboardInterrupt:
        print("\n[ÇIKIŞ] Kullanıcı gösterim kaydını durdurdu.")
    finally:
        h5_file.close()
        controller.close()
        env.close()
        print(f"\n[BİLGİ] Toplam {saved_episodes} başarılı epizot '{output_path}' konumuna kaydedildi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ManiSkill Gösterim Kaydedici")
    parser.add_argument("--env-id", type=str, default="MahutKol-PickCube-v1", help="Ortam kimliği")
    parser.add_argument("--output-path", type=str, default="data/demos/raw_demos.h5", help="Kayıt H5 yolu")
    parser.add_argument("--num-episodes", type=int, default=50, help="Toplanacak başarılı bölüm sayısı")
    parser.add_argument("--device", type=str, default="gamepad", choices=["keyboard", "gamepad", "ps4", "spacemouse", "vr"])
    parser.add_argument("--freq", type=int, default=20, help="Kontrol frekansı (Hz)")
    args = parser.parse_args()

    record_demonstrations(
        env_id=args.env_id,
        output_path=args.output_path,
        num_episodes=args.num_episodes,
        device_type=args.device,
        control_freq=args.freq,
    )

