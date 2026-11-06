# 🤗 Hugging Face LeRobot Geliştirici ve Mimari Kılavuzu

**LeRobot**, PyTorch tabanlı ve Hugging Face ekosistemiyle tam entegre, modern robotik taklit öğrenme (Imitation Learning / Behavioral Cloning) ve robot politikası eğitim kütüphanesidir.

* **Resmi Dokümantasyon:** [https://huggingface.co/docs/lerobot](https://huggingface.co/docs/lerobot)
* **Kaynak Kod (GitHub):** [https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
* **Politikalar Kılavuzu:** [LeRobot Policies](https://huggingface.co/docs/lerobot/main/en/policies/index.html)
  * [ACT (Action Chunking Transformer)](https://huggingface.co/docs/lerobot/main/en/policies/act.html)
  * [Diffusion Policy](https://huggingface.co/docs/lerobot/main/en/policies/diffusion.html)
* **Veri Seti (Datasets) Kılavuzu:** [LeRobotDataset](https://huggingface.co/docs/lerobot/main/en/advanced_guides/datasets.html)

---

## 1. Temel Bileşenler ve Mimari

LeRobot 3 temel ayak üzerine kuruludur:
1. **Veri Seti Standardı (`LeRobotDataset`):** Çok modlu (video kamera görüntüleri, eklem durumları, motor eylemleri) veriyi standart Parquet + MP4 / Zarr formatında optimize depolar.
2. **Politikalar (Model Zoo):**
   * **ACT:** Hızlı eğitim (1-3 saat), az gösterimle yüksek başarı, transformatör tabanlı eylem yığınlama (action chunking).
   * **Diffusion Policy:** Karmaşık temas içeren işler, multimodal eylem dağılımları ve yüksek hassasiyet.
3. **Eğitim ve Değerlendirme:** CLI komutları (`lerobot.scripts.train`, `lerobot.scripts.eval`) ve Python API.

---

## 2. LeRobot Veri Seti Mimarisi (`LeRobotDataset`)

ManiSkill'de toplanan simülasyon kayıtlarını LeRobot formatına dönüştürmek için kullanılan temel API şeması:

```python
from lerobot.datasets import LeRobotDataset

# 1. Veri seti şemasını başlat
dataset = LeRobotDataset.create(
    repo_id="local/mahut_kol_pick_cube",
    fps=20,
    robot_type="mahut_kol",
    features={
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),  # 6 kol eklemi + 2 parmak pozisyonu
            "names": ["q1", "q2", "q3", "q4", "q5", "q6", "gripper_l", "gripper_r"],
        },
        "observation.images.base_camera": {
            "dtype": "image",
            "shape": (3, 128, 128),  # (C, H, W)
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_camera": {
            "dtype": "image",
            "shape": (3, 128, 128),
            "names": ["channels", "height", "width"],
        },
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["dq1", "dq2", "dq3", "dq4", "dq5", "dq6", "cmd_l", "cmd_r"],
        },
    },
)

# 2. Epizot ve Kareleri Ekle
for episode_data in all_episodes:
    for frame in episode_data:
        dataset.add_frame({
            "observation.state": frame["state"],
            "observation.images.base_camera": frame["base_img"],
            "observation.images.wrist_camera": frame["wrist_img"],
            "action": frame["action"],
        })
    dataset.save_episode()

# 3. Diske Yaz ve Optimize Et
dataset.finalize()
```

---

## 3. Politika Eğitimi (Training)

### A. CLI ile Doğrudan Çalıştırma
```bash
python -m lerobot.scripts.train \
    --policy.type=act \
    --dataset.repo_id=local/mahut_kol_pick_cube \
    --output_dir=outputs/train_act \
    --policy.chunk_size=50 \
    --policy.n_action_steps=50 \
    --policy.dim_model=512 \
    --training.batch_size=32 \
    --training.steps=50000 \
    --training.lr=1e-5 \
    --policy.device=cuda
```

### B. Python API ile Eğitim Döngüsü
```python
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig

# 1. Konfigürasyon ve Model
config = ACTConfig(
    chunk_size=50,
    n_action_steps=50,
    input_shapes={
        "observation.images.base_camera": [3, 128, 128],
        "observation.state": [8],
    },
    output_shapes={"action": [8]},
)
policy = ACTPolicy(config)
policy.to("cuda")

# 2. Veri Yükleyici (DataLoader)
dataset = LeRobotDataset("local/mahut_kol_pick_cube")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Optimize Edici
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, weight_decay=1e-4)

# 4. Eğitim Adımı
policy.train()
for step, batch in enumerate(dataloader):
    # Cihaza taşı
    batch = {k: v.to("cuda") for k, v in batch.items()}
    loss, loss_dict = policy.forward(batch)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

---

## 4. Simülasyonda Kapalı Çevrim Değerlendirme (Closed-Loop Evaluation)

Eğitilen politika simülasyon ortamına bağlanırken **Temporal Ensembling** ve **Action Queue** kullanılır:

```python
import gymnasium as gym
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy

# 1. Eğitilmiş Modeli Yükle
policy = ACTPolicy.from_pretrained("outputs/train_act/checkpoints/last/pretrained_model")
policy.eval().to("cuda")

# 2. Ortamı Başlat
env = gym.make("MahutKol-PickCube-v0", obs_mode="rgbd")

# 3. Bölüm Döngüsü
obs, info = env.reset()
policy.reset()  # Aksiyon tamponunu sıfırla

done = False
while not done:
    # Gözlemi tensöre çevir
    batch = {
        "observation.state": torch.from_numpy(obs["state"]).unsqueeze(0).to("cuda"),
        "observation.images.base_camera": torch.from_numpy(obs["image_base"]).permute(2,0,1).unsqueeze(0).to("cuda"),
    }
    
    with torch.no_grad():
        action = policy.select_action(batch)
        
    obs, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
    done = terminated or truncated

print("Görev Başarısı:", info["success"])
```

---

## 5. Donanım ve Bellek İpuçları

| Politika | Minimum VRAM | Önerilen GPU | Eğitim Süresi (~50 demo) |
| :--- | :--- | :--- | :--- |
| **ACT** | 6 - 8 GB | RTX 3060 / 3070 / 4070 | 1 - 2.5 Saat |
| **Diffusion Policy** | 10 - 16 GB | RTX 3090 / 4090 / A5000 | 4 - 8 Saat |
