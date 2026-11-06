"""
Eğitilmiş Politikanın Simülasyonda Değerlendirilmesi (Closed-Loop Evaluation).

Bu betik:
1. Eğitilmiş LeRobot modelini (ACT / Diffusion) yükler.
2. ManiSkill ortamını ("MahutKol-PickCube-v1") başlatır.
3. Kapalı çevrimde (Closed-loop) çalışır:
   - Ortamdan gözlemi alır (Kamera RGB + Proprioception state)
   - Modele besler (`policy.select_action(batch)`)
   - Üretilen eylemi simülasyona uygular (`env.step(action)`)
4. Belirlenen bölüm sayısı (örn: 50 bölüm) üzerinden Genel Başarı Oranını (% Success Rate) raporlar.
"""

import argparse
from typing import Dict, Any
import torch


def evaluate_policy(
    checkpoint_path: str,
    env_id: str = "MahutKol-PickCube-v1",
    num_eval_episodes: int = 50,
    device: str = "cuda"
) -> float:
    """
    Kapalı çevrim değerlendirme döngüsü.

    Çıktı:
    - success_rate (float): 0.0 ile 1.0 arasında başarı yüzdesi.

    TODO Adımları:
    1. Eğitilmiş modeli `lerobot` üzerinden yükleyin (`policy.from_pretrained(checkpoint_path)`).
    2. `policy.eval()` ve `.to(device)` durumuna getirin.
    3. `gymnasium.make(env_id, obs_mode="rgbd", render_mode="rgb_array")` ile ortamı açın.
    4. Başarı sayacı (success_count = 0) tutun.
    5. Her bölüm için:
       - obs, _ = env.reset()
       - policy.reset()  # Temporal ensembling / action queue sıfırlaması
       - while not done:
           - Gözlemleri tensöre çevirip batch formatına sokun:
             batch = {
                 "observation.state": torch.from_numpy(obs['state']).to(device),
                 "observation.images.base_camera": torch.from_numpy(obs['image']).to(device)
             }
           - action = policy.select_action(batch)
           - obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
       - if info['success']: success_count += 1
    6. `success_rate = success_count / num_eval_episodes` hesaplayıp ekrana yazdırın.
    """
    raise NotImplementedError("TODO: evaluate_policy() döngüsünü doldurun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ManiSkill Kapalı Çevrim Değerlendirici")
    parser.add_argument("--checkpoint", type=str, required=True, help="Eğitilmiş LeRobot model klasörü")
    parser.add_argument("--env-id", type=str, default="MahutKol-PickCube-v1")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate_policy(
        checkpoint_path=args.checkpoint,
        env_id=args.env_id,
        num_eval_episodes=args.num_episodes,
        device=args.device,
    )
