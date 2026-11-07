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
[Komut: Türkçe ses/metin → CommandParser JSON → Command (src/mahut_kol/commands/)]
       ↓ reset(options) + sabit İngilizce talimat
[Telefon (WebXR) / PS4 / klavye teleop (scripts/record_vla_demos.py)]
[+ Otomatik scripted demolar (scripts/generate_scripted_demos.py)]
       ↓ LeRobotDataset v3 (lerobot 0.4.4) + mahut_episodes.jsonl
[Doğrulama + kapsama raporu (scripts/validate_dataset.py)]
       ↓
[LeRobot ile Politika Eğitimi (scripts/train_smolvla.sh; ileride Modal)]
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
│   └── commands/heldout_commands.json # Değerlendirmeye ayrılan 30 komut (sabit)
├── docs/                     # Ayrıntılı geliştirici kılavuzları
│   ├── DONANIM_BILGILERI.md  # Feetech FT5330M, ESP32-S3 ve mekanik hesaplar
│   ├── MANISKILL_GUIDE.md    # ManiSkill 3 geliştirici referansı
│   ├── LEROBOT_GUIDE.md      # LeRobot eğitim ve veri standardı referansı
│   ├── MAHUTKOL_PIPELINE.md  # Entegrasyon hattı teknik şeması
│   ├── VLA_GOREV_TANIMLARI.md # Yön tanımları, başarı kriteri, kameralar, çalışma alanı
│   ├── workspace_map.txt     # Çalışma alanı FK/IK örnekleme çıktısı
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
│   ├── collect_demos.py      # (eski) PickCube scripted veri toplama
│   ├── record_vla_demos.py   # VLA teleop kaydı (telefon/PS4/klavye; queue/text/voice komut)
│   ├── generate_scripted_demos.py # Otomatik scripted VLA demoları (aynı şema)
│   ├── validate_dataset.py   # Veri kümesi doğrulama + kapsama + held-out sızıntı raporu
│   ├── compute_workspace.py  # Erişilebilir çalışma alanını FK/IK ile örnekler
│   ├── render_task_cameras.py# front/wrist kameralarını render edip yön tanımlarını doğrular
│   ├── make_heldout_split.py # Held-out komut listesini (bir kez) üretir
│   ├── train_smolvla.sh      # LeRobot SmolVLA eğitim başlatıcı
│   ├── eval_policy.py        # Çok şekilli görevde kapalı çevrim değerlendirme (dil testi dahil)
│   ├── check_urdf.py         # URDF doğrulama aracı
│   └── test_ps4.py           # (Teleoperasyonlar/ yönlendiricisi)
├── src/mahut_kol/
│   ├── robots/               # MahutRobot (BaseAgent) + kinematics.py (URDF FK/IK)
│   ├── envs/                 # Görev ortamları; multi_shape_task.py + scripted_policy.py
│   ├── commands/             # Komut şeması, İngilizce talimat, slotlar, dengeli kuyruk, held-out
│   ├── recording/            # LeRobot yazıcı (devam/ekleme), sidecar, kayıt döngüsü/arayüz
│   ├── teleop/               # ActionSource arayüzü, telefon WebXR (phone_webxr.py + web/index.html)
│   └── speech/               # Yerel Türkçe ses -> komut hattı
│   (eski datasets/, train/, eval/ stub paketleri silindi; yeni kodu `datasets/` adlı
│    klasöre koymayın, .gitignore yok sayar)
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

### 7. Eğitilen Modeli Simülasyonda Değerlendirme (çok şekilli görev):
```bash
~/.venvs/lerobot/bin/python scripts/eval_policy.py --checkpoint latest --episodes 45 --heldout-episodes 30 --language-scenes 10
# model olmadan boru hattı duman testi:
~/.venvs/lerobot/bin/python scripts/eval_policy.py --policy scripted --episodes 3 --heldout-episodes 2 --language-scenes 1
```
> Not: `scripts/train_smolvla.sh` hâlâ eski PickCube veri kümesine (`~/vla_data/pick_cube_arm`)
> göre yazılı; yeni veri kümesiyle eğitim (`~/vla_data/mahut_multishape`, kameralar
> `observation.images.front` + `observation.images.wrist`) bir sonraki aşamanın işidir.

---

## 7. ⚠️ Önemli Kurulum Notu
Proje, simülasyon ve görüntü işleme kütüphanelerinin C-API uyumluluğu nedeniyle **NumPy 1.x** (`numpy<2`, örn. `1.26.4`) gerektirir:
```bash
pip install "numpy<2"
```

---

## 8. 🗣️ VLA Veri Toplama Hattı (Çok Şekilli İlişkisel Yerleştirme)

Ayrıntılı tanımlar: [docs/VLA_GOREV_TANIMLARI.md](docs/VLA_GOREV_TANIMLARI.md).

### Komut ve veri akışı
```text
Türkçe ses ─▶ VoiceCommander (faster-whisper) ─┐
Türkçe metin ──────────────────────────────────┼─▶ CommandParser JSON ─▶ Command
Dengeli komut kuyruğu (varsayılan) ────────────┘        │
                                                         ├─▶ env.reset(options)  (MahutKol-MultiShapeRelational-v1)
                                                         ├─▶ İngilizce talimat (sabit şablon + seed'li paraphrase)  → LeRobot `task`
                                                         └─▶ 5 sayısal slot (renk, şekil, renk, şekil, yön)        → `command.slots`
```
Eğitime giren metin her zaman İngilizce şablondur; ham Türkçe/STT metni yalnızca sidecar'da
(`command_input`) saklanır. Geçersiz/onay bekleyen komutta bölüm başlatılmaz.

### Yön tanımları (front kamera görüntüsüne göre)
Robot +x'e bakar; sabit kamera operatör tarafında (x > 0) robota bakar → **görüntünün sağı = +y**.
FRONT = referansın robota bakan tarafı (−x), BACK = kameraya bakan tarafı (+x), LEFT = −y,
RIGHT = +y, TOP = üstü (referans üçgen prizmaysa geçersiz). Δ = 0.06 m. 336 geçerli komut.

### Kameralar (geçici; donanım netleşince güncellenecek)
* **front:** sabit, operatör tarafında, 224×224.
* **wrist:** kıskacın üstüne (TCP) monte, parmak uçlarına bakar, 224×224.
Değerler `src/mahut_kol/envs/multi_shape_task.py` başındaki `CAMERA_*` sabitlerinde tek yerden değişir.

### Veri şeması (LeRobotDataset, lerobot 0.4.4, fps 20)
| Özellik | Tür | Şekil |
|---|---|---|
| `observation.images.front`, `observation.images.wrist` | video | 224×224×3 |
| `observation.state` | float32 | 7 (joint0-4, finger_left, finger_right) |
| `action` | float32 | 7 (mutlak `pd_joint_pos` hedefi) |
| `command.slots` | int64 | 5 |
| `task` | metin | İngilizce talimat |

Varsayılan kök `~/vla_data/mahut_multishape`, repo_id `local/mahut_multishape` (`--root`, `--repo-id`).
Var olan veri kümesi her zaman açılıp sonuna eklenir; hiçbir şey silinmez; şema uyuşmazsa hata.
Sidecar `<root>/mahut_episodes.jsonl`: bölüm indeksi, seed, komut JSON'u, talimat, slotlar,
held-out bayrağı, `source: teleop|scripted`, başarı, süre.

### Komutlar
```bash
# Teleop (telefon). Önce telefonu USB ile bağlayın:
adb reverse tcp:8000 tcp:8000
~/.venvs/lerobot/bin/python scripts/record_vla_demos.py --device phone --num-episodes 20
# PS4 / klavye
~/.venvs/lerobot/bin/python scripts/record_vla_demos.py --device gamepad
# Komutu Türkçe yazarak ya da sesle vermek
~/.venvs/lerobot/bin/python scripts/record_vla_demos.py --command-source text
~/.venvs/lerobot/bin/python scripts/record_vla_demos.py --command-source voice --voice-mode push_to_talk
# Otomatik scripted demolar (aynı veri kümesine eklenir, kaldığı yerden devam eder)
~/.venvs/lerobot/bin/python scripts/generate_scripted_demos.py --num-episodes 2000
# Doğrulama + kapsama raporu
~/.venvs/lerobot/bin/python scripts/validate_dataset.py --video-check sample
```
Kayıt penceresinde: **Enter** kaydet (başarı otomatik algılandıktan sonra), **R** at ve yeniden
dene, **Q** çık. Held-out komutlar kayıtta atlanır (`--include-heldout` ile açılır).

### Telefonla teleop (WebXR) – adım adım deneme
1. Android telefonda Geliştirici seçenekleri → **USB hata ayıklama** açık; telefonda **Google Play
   Services for AR (ARCore)** kurulu olmalı.
2. Telefonu USB ile bağlayın, `adb devices` telefonu göstermeli.
3. `adb reverse tcp:8000 tcp:8000` (telefondaki localhost:8000 → bilgisayar).
4. `~/.venvs/lerobot/bin/python scripts/record_vla_demos.py --device phone --num-episodes 3`
5. Telefonda Chrome ile `http://localhost:8000` açın; üstte "Bağlandı (ManiSkill)" görünmeli,
   "Görev" kutusunda İngilizce talimat yazmalı.
6. **Bilgisayar ekranına (robota) bakarken** "AR Teleop Başlat"a dokunun (eksenler bu andaki
   yönelime sabitlenir).
7. **DEBRİYAJ** düğmesini basılı tutup telefonu hareket ettirin: ileri (ekrana doğru) → kıskaç
   robota doğru (−x), sağa → +y (görüntüde sağa), yukarı → yukarı. Telefonun burnunu aşağı
   eğmek kıskacı dikleştirir; direksiyon gibi çevirmek bilek roll'ü. Düğmeyi bırakınca kol durur
   (yeniden konumlanmak için bırakıp telefonu rahat pozisyona getirin).
8. Kaydırıcı: %100 tam açık, %0 kapalı. Nesneyi kavrayıp hedefe koyun, kaydırıcıyı açarak bırakın.
9. Başarı algılanınca pencere yeşile döner: **Enter** kaydet, **R** at. Telefondaki "Sıfırla"
   düğmesi bölümü atıp yeniden başlatır.
10. Ters giden bir eksen varsa `src/mahut_kol/teleop/phone_webxr.py` içindeki `CONFIG["axis_signs"]`
    / `tilt_from` / `roll_from` işaretlerini değiştirin.

### Çalışma alanı ve kavrama
MahutKol tam aşağı bakarak yalnızca tabana çok yakın erişebildiği için kavrama düşeyden 45–55°
eğik yapılır. Nesne/hedef alanı x ∈ [0.10, 0.22], y ∈ [−0.16, 0.12] m (`scripts/compute_workspace.py`,
çıktı `docs/workspace_map.txt`). Kinematik `src/mahut_kol/robots/kinematics.py`: URDF'ten saf numpy
FK (SAPIEN TCP pozuyla birebir), eğim kısıtlı DLS IK; telefon teleopu ve scripted demolar aynı
çözücüyü kullanır.

