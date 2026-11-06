"""
LeRobot Politika Eğitim Başlatıcı Şablonu.

Bu betik, dönüştürülen `LeRobotDataset` üzerinde ACT (Action Chunking with Transformers)
veya Diffusion Policy modellerini eğitmek için köprü görevi görür.

LeRobot CLI Kullanım Alternatifi:
```bash
python -m lerobot.scripts.train \
    --policy.type=act \
    --dataset_repo_id=local/mahut_kol_pick_cube \
    --output_dir=outputs/train_act \
    --training.batch_size=32 \
    --training.steps=50000
```
"""

import argparse
from typing import Optional
import yaml


def train_policy(config_path: str):
    """
    YAML konfigürasyonunu okuyarak LeRobot eğitim döngüsünü başlatır.

    TODO Adımları:
    1. `yaml.safe_load` ile konfigürasyonu okuyun.
    2. `lerobot.common.policies.act.modeling_act.ACTPolicy` veya `DiffusionPolicy` yapılandırmasını başlatın.
    3. `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` üzerinden veri yükleyiciyi (DataLoader) oluşturun.
    4. Optimizer (AdamW), LR Scheduler ve Loss fonksiyonunu (L1 + KL Loss) bağlayın.
    5. Eğitim döngüsünü (Training loop) çalıştırıp checkpoint'leri `output_dir` içine kaydedin.
    """
    raise NotImplementedError("TODO: train_policy() fonksiyonunu LeRobot API'sine göre doldurun.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeRobot Eğitim Başlatıcı")
    parser.add_argument("--config", type=str, default="configs/train/lerobot_act.yaml")
    args = parser.parse_args()

    train_policy(config_path=args.config)
