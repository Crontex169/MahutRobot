# 📖 MahutKol: Proje Açıklama ve Kullanım Kılavuzu

---

## 1. 🎯 Projenin Amacı ve Vizyonu

**MahutKol**, özel tasarım 5-DOF kol ve çift parmak paralel kıskaçtan oluşan robotik kol mimarisi üzerine kurulu, uçtan uca modern bir **Robot Öğrenme (Robot Learning)** altyapısıdır.

Projenin temel hedefi:
1. Özel robot kolunu fizik tabanlı simülasyon ortamında (**ManiSkill 3 / SAPIEN / PhysX**) çalıştırmak,
2. İnsan gösterimlerini (demonstrations) **PS4 Kolu (Gamepad)**, **VR (Meta Quest)** veya **Klavye** aracılığıyla yüksek kalitede toplamak,
3. Toplanan gösterimleri Hugging Face **LeRobot** formatına dönüştürmek,
4. **SmolVLA**, **ACT (Action Chunking with Transformers)** veya **Diffusion Policy** gibi modern taklit öğrenme (Imitation Learning) modelleri ile eğitmek,
5. Eğitilen modeli simülasyonda kapalı çevrimde (closed-loop) test edip başarı oranını ölçmektir.

---

## 2. 🦾 Robot Modeli ve Donanım Spesifikasyonu

Projede kullanılan robot kolu, 3D baskı parçalar, yüksek torklu **Feetech FT5330M 35Kg (7.4V)** dijital servo motorlar ve **ESP32-S3-N16R8** kontrolcü kartı ile çalışan 5-DOF kol ve çift parmak paralel kıskaçtan oluşur:

| Eklem No | Eklem Adı | Motor Tipi | Stall Torku | Hareket Açıklaması |
| :---: | :--- | :--- | :--- | :--- |
| **0** | `joint0` | Feetech FT5330M | 35 kg·cm (3.432 Nm) | 100mm Lazy Susan rulmanlı taban dönüşü (Yaw) |
| **1** | `joint1` | Feetech FT5330M | 35 kg·cm (3.432 Nm) | Omuz eğim eklemi (Shoulder Pitch) |
| **2** | `joint2` | Feetech FT5330M | 35 kg·cm (3.432 Nm) | Dirsek eğim eklemi (Elbow Pitch) |
| **3** | `joint3` | Feetech FT5330M | 35 kg·cm (3.432 Nm) | Bilek eğim eklemi (Wrist Pitch) |
| **4** | `joint4` | Feetech FT5330M | 35 kg·cm (3.432 Nm) | Bilek dönüş eklemi (Wrist Roll) |
| **5-6** | `finger_left/right` | Feetech FT5330M | 35 kg·cm (15N tutuş) | Çift parmaklı lineer kramayer kıskaç (Gripper-v2) |

> 📌 **Ayrıntılı Donanım ve Sim-to-Real Kılavuzu:** Motor torkları, ESP32-S3 mimarisi ve mekanik hesaplar için [docs/DONANIM_BILGILERI.md](docs/DONANIM_BILGILERI.md) belgesine bakınız.

---

## 3. 🔄 Uçtan Uca Boru Hattı (Workflow Architecture)

```text
[CAD Tasarımı (assets/meshes/)]
       ↓
[URDF Modeli (assets/urdf/mahut_kol.urdf)]
       ↓
[ManiSkill Simülasyonu (src/mahut_kol/robots & envs)]
       ↓
[PS4 / Gamepad ile Teleoperasyon (Teleoperasyonlar/teleop_maniskill_ps4.py)]
       ↓ (LeRobotDataset / HDF5)
[LeRobot ile Politika Eğitimi (scripts/train_smolvla.sh)]
       ↓ (Model Checkpoint)
[Simülasyonda Kapalı Çevrim Test (scripts/eval_policy.py)]
```

---

## 4. 📁 Klasör ve Dosya Hiyerarşisi

```text
MahutKol/
├── assets/
│   ├── meshes/               # Robotun 3D STL parçaları (motorlar, gövde, kıskaç)
│   ├── urdf/mahut_kol.urdf   # MahutKol eklem, link ve atalet tanımları
│   ├── mahut_kol_robot.xml   # Bağımsız MuJoCo robot modeli
│   ├── mahut_kol_maniskill.xml # ManiSkill için optimize MJCF modeli
│   └── mahut_kol_scene.xml   # Masa, küp ve kameralı tam MuJoCo sahnesi
├── configs/
│   ├── robot/mahut_kol.yaml  # Eklem adları, PD kazançları, qpos sınırları
│   ├── task/pick_place.yaml  # Küp taşıma görevi ve kamera parametreleri
│   └── train/lerobot_act.yaml# LeRobot eğitim hiperparametreleri (ACT)
├── docs/                     # Ayrıntılı geliştirici kılavuzları
│   ├── DONANIM_BILGILERI.md  # Feetech FT5330M, ESP32-S3 ve mekanik hesaplar
│   ├── MANISKILL_GUIDE.md    # ManiSkill 3 geliştirici referansı
│   ├── LEROBOT_GUIDE.md      # LeRobot eğitim ve veri standardı referansı
│   ├── MAHUTKOL_PIPELINE.md  # Entegrasyon hattı teknik şeması
│   ├── VR_TELEOPERATION_GUIDE.md # Meta Quest VR teleoperasyon kılavuzu
│   └── GAMEPAD_TELEOP_GUIDE.md   # PS4 kolu tuş haritası
├── Teleoperasyonlar/         # 🎮 Teleoperasyon Kontrolcüleri ve Betikleri
│   ├── teleop_maniskill_ps4.py # PS4 / Gamepad ile MahutKol teleoperasyon betiği
│   ├── test_ps4.py           # Gamepad donanım test betiği
│   ├── record_demos.py       # Teleoperasyonla gösterim kaydedici
│   ├── teleop_agent.py       # Kartezyen teleoperasyon kontrolcüsü
│   ├── teleop_ps4.py         # 7-DoF eklem teleoperasyon kontrolcüsü
│   └── README.md             # Teleoperasyon rehberi ve tuş haritası
├── scripts/
│   ├── teleop_maniskill_ps4.py # (Teleoperasyonlar/ yönlendiricisi)
│   ├── sim_custom_task.py    # PushCube özel görev görselleştirici
│   ├── sim_minimal.py        # Sade sahne simülatörü
│   ├── view_mujoco.py        # MuJoCo model ve sahne görüntüleyici
│   ├── collect_demos.py      # Otonom / teleop veri toplama aracı
│   ├── train_smolvla.sh      # LeRobot SmolVLA eğitim başlatıcı
│   ├── eval_policy.py        # Kapalı çevrim model başarı değerlendirici
│   ├── check_urdf.py         # URDF doğrulama aracı
│   └── test_ps4.py           # (Teleoperasyonlar/ yönlendiricisi)
├── src/mahut_kol/
│   ├── robots/               # ManiSkill BaseAgent robot sınıfları (MahutRobot)
│   ├── envs/                 # Küp taşıma/itme görev ortamları (Gymnasium)
│   ├── teleop/               # PS4 ve teleoperasyon ajanları
│   ├── datasets/             # HDF5 ➔ LeRobotDataset dönüştürücü
│   ├── train/                # Politika eğitim başlatıcı
│   └── eval/                 # Kapalı çevrim başarı oranı değerlendirici
├── requirements.txt          # Python paket bağımlılıkları (numpy<2 vb.)
├── pyproject.toml            # Paket kurulum yapılandırması
├── YAPILANLAR.md             # Kronolojik geliştirme günlüğü
└── PROJE_REHBERI.md          # Bu genel açıklama dosyası
```

---

## 5. 🎮 Teleoperasyon ve Kontrol Seçenekleri

### A. PS4 Kolu (DualShock 4) ile Kontrol (MahutKol):
* **Sol Analog:** Taban Dönüşü (`joint0` Yaw) ve Omuz Eğimi (`joint1` Pitch)
* **Sağ Analog:** Dirsek Eğimi (`joint2` Pitch) ve Bilek Eğimi (`joint3` Pitch)
* **D-Pad Sol/Sağ:** Bilek Yuvarlanma (`joint4` Roll)
* **R2 / L2 Tetikleri:** Kıskaç Kapat / Aç (`finger_left`, `finger_right`)
* **Üçgen / Options:** Epizotu Sıfırla (Reset)
* **Klavye:** W/S (Omuz), A/D (Taban), Yön Tuşları (Dirsek/Bilek), Boşluk (Kıskaç Kapat), F (Kıskaç Aç)

### B. VR Başlığı (Meta Quest) ile Kontrol:
* Operatörün el hareketi uzayda 6-DoF olarak takip edilir ve Ters Kinematik (IK) ile robotun uç organına haritalanır.
* Detaylar için: [docs/VR_TELEOPERATION_GUIDE.md](docs/VR_TELEOPERATION_GUIDE.md).

---

## 6. 🚀 Hızlı Başlangıç ve Komut Kılavuzu

### 1. MahutKol Özel Modelini Simülasyonda PS4 / Gamepad ile Kontrol Etme:
```bash
python Teleoperasyonlar/teleop_maniskill_ps4.py
```
*(Varsayılan olarak `MahutKol-PickCube-v1` ortamında kullanıcının özel 5-DOF + paralel kıskaçlı modeli açılır)*

### 2. Özel Görev (PushCube) Ortamını Çalıştırma:
```bash
python scripts/sim_custom_task.py
```

### 3. MuJoCo İnteraktif Sahne ve Robot Görüntüleyici:
```bash
python scripts/view_mujoco.py
```

### 4. URDF Doğrulama ve Eklem Limitleri Testi:
```bash
python scripts/check_urdf.py
```

### 5. Eğitim İçin Gösterim (Demonstration) Toplama:
```bash
python scripts/collect_demos.py 50
```

### 6. SmolVLA / LeRobot Model Eğitimi:
```bash
bash scripts/train_smolvla.sh 5000 8
```

### 7. Eğitilen Modeli Simülasyonda Değerlendirme:
```bash
python scripts/eval_policy.py latest 20
```

---

## 7. ⚠️ Önemli Kurulum Notu
Proje, simülasyon ve görüntü işleme kütüphanelerinin C-API uyumluluğu nedeniyle **NumPy 1.x** (`numpy<2`, örn. `1.26.4`) gerektirir:
```bash
pip install "numpy<2"
```
