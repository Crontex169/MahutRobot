# 🔄 MahutKol: ManiSkill & LeRobot Entegrasyon Hattı

Bu doküman, **ManiSkill 3** simülasyon ortamı ile **Hugging Face LeRobot** eğitim çerçevesinin projemizde nasıl birlikte çalıştığını ve veri akışının adımlarını açıklar.

---

## 🏗️ Uçtan Uca Mimari ve Veri Akışı

```text
┌─────────────────────────┐
│ CAD Tasarımı (STL/STP)   │ ──▶ assets/tasarım/ (22 STL parçası)
└────────────┬────────────┘
             ▼ (CoACD + Trimesh)
┌─────────────────────────┐
│ URDF + Meshler           │ ──▶ assets/urdf/mahut_kol.urdf & assets/meshes/
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ ManiSkill Robot & Görev │ ──▶ src/mahut_kol/robots/mahut_robot.py
│ Simülasyon Ortamı       │ ──▶ src/mahut_kol/envs/pick_place_task.py
└────────────┬────────────┘
             ▼ (Teleoperasyon / Klavye / Gamepad)
┌─────────────────────────┐
│ Ham Simülasyon Verisi   │ ──▶ src/mahut_kol/teleop/record_demos.py (HDF5 formatı)
│ (Demos / Trajectories)  │
└────────────┬────────────┘
             ▼ (convert_to_lerobot.py)
┌─────────────────────────┐
│ LeRobot Veri Seti       │ ──▶ src/mahut_kol/datasets/convert_to_lerobot.py
│ (LeRobotDataset v3)     │     (Parquet + MP4 / Zarr)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Model Eğitimi           │ ──▶ src/mahut_kol/train/train_policy.py
│ (ACT / Diffusion)       │     (configs/train/lerobot_act.yaml)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Kapalı Çevrim Test      │ ──▶ src/mahut_kol/eval/evaluate_policy.py
│ (Closed-Loop Eval)      │     (ManiSkill içinde başarı oranı % ölçümü)
└─────────────────────────┘
```

---

## 📂 Kod Dosyaları ve Kütüphane Rolleri

| Dosya Yolu | Sorumlu Kütüphane | Rol ve İşlev |
| :--- | :--- | :--- |
| `src/mahut_kol/robots/mahut_robot.py` | **ManiSkill 3** | Robotu `BaseAgent` olarak tanımlar, eklemleri eşleştirir, PD kazançlarını belirler. |
| `src/mahut_kol/envs/pick_place_task.py` | **ManiSkill 3** | Görevin masasını, taşınacak küpü, başarı kriterlerini ve kameraları kurar. |
| `src/mahut_kol/teleop/record_demos.py` | **ManiSkill + Pygame** | Simülasyonu çalıştırıp kullanıcı girişleriyle başarılı bölümleri HDF5'e yazar. |
| `src/mahut_kol/datasets/convert_to_lerobot.py` | **LeRobot** | HDF5 verisini `LeRobotDataset` formatına dönüştürür. |
| `src/mahut_kol/train/train_policy.py` | **LeRobot** | ACT veya Diffusion Policy modelini LeRobot motoruyla eğitir. |
| `src/mahut_kol/eval/evaluate_policy.py` | **ManiSkill + LeRobot** | Eğitilmiş ağırlıkları ManiSkill ortamında test edip başarı yüzdesini çıkarır. |

---

## 🔗 İlgili Ayrıntılı Kılavuzlar
* [ManiSkill 3 Kılavuzu](file:///home/deniz/Ortak/MahutKol/docs/MANISKILL_GUIDE.md)
* [LeRobot Kılavuzu](file:///home/deniz/Ortak/MahutKol/docs/LEROBOT_GUIDE.md)
