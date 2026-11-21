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

Projede kullanılan robot kolu (yeni kol, 2026-11), 3D baskı parçalar ve 6 adet **Feetech FT5330M**
(PWM 500–2500 µs, ±90°, ölü bant ~0.4°, mekanik durdurucu yok) servo ile çalışan 5-DOF kol ve tek
servolu çift parmak paralel kıskaçtan oluşur. Model kaynağı `Yeni_model/arm_mujoco.xml` (Fusion
MJCF, git'te izlenmez); URDF `scripts/mjcf_to_urdf.py` ile üretilir.

| Eklem | Ad (MJCF) | Hareket | Aralık |
| :---: | :--- | :--- | :--- |
| 0 | `joint0` (j1) | taban yaw | ±90° |
| 1 | `joint1` (j2) | omuz pitch | ±90° |
| 2 | `joint2` (j3) | dirsek pitch | ±90° |
| 3 | `joint3` (j4) | bilek pitch | ±90° |
| 4 | `joint4` (j5) | bilek roll (= yaklaşma ekseni) | ±90° |
| kıskaç | servo j6 → `finger_left/right` | tek servo, iki prismatik parmak | 0 = açık (46 mm), π rad = kapalı |

**CAD'den bilinçli sapmalar (önemli):** simülasyonda kıskaç roll servosuna **90° döndürülmüş**
monte edilmiştir (`joint4 = 0`'da parmaklar yatay; CAD'de kol düzlemindeydi) ve açık kıskaç
**46 mm**'dir (CAD 38 mm). Gerçek kol buna göre kurulmalı; ayrıntı `scripts/mjcf_to_urdf.py`.

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
[Modal'da politika eğitimi (cloud/mahut_cloud.py; SmolVLA / ACT)]
       ↓ (Model Checkpoint)
[Simülasyonda Kapalı Çevrim Test (scripts/eval_policy.py)]
```

---

## 4. 📁 Klasör ve Dosya Hiyerarşisi

Tüm yollar `Main_project/` altındadır; komutlar bu klasörden çalıştırılır.

```text
Main_project/
├── assets/
│   ├── meshes/               # Yeni kolun STL parçaları (mjcf_to_urdf.py üretir)
│   ├── urdf/mahut_kol.urdf   # Üretilmiş URDF (elle düzenleme; kaynak Yeni_model/arm_mujoco.xml)
│   └── objects/              # Görev nesne meshleri (üçgen prizma)
├── cloud/                    # Modal eğitim hattı (mahut_cloud.py) + eğitim devri (devir.py)
├── configs/
│   ├── robot/mahut_kol.yaml  # Robot özeti (kod okumaz; tek kaynak mahut_robot.py)
│   ├── training/             # smolvla.yaml, act.yaml (koşu ayarları + maliyet modeli)
│   └── commands/heldout_commands.json # Değerlendirmeye ayrılan 30 komut (sabit)
├── Data_Model/               # Eğitim devri tar'ları (git'e girmez, yalnızca README)
├── docs/
│   ├── DONANIM_BILGILERI.md  # Feetech FT5330M, ESP32-S3 ve mekanik hesaplar
│   ├── VLA_GOREV_TANIMLARI.md # Yön tanımları, başarı kriteri, kameralar, çalışma alanı
│   ├── EGITIM_DEVIR_KILAVUZU.md # Arkadaşlar için eğitim devri kılavuzu
│   ├── workspace_map.txt     # Çalışma alanı FK/IK örnekleme çıktısı
│   └── yeni_kol/             # Kamera görüntüleri (render_task_cameras.py)
├── scripts/                  # Yalnızca giriş noktaları; kayıt/doğrulama/eval/prizma betikleri src'deki main()'i çağırır
│   ├── mjcf_to_urdf.py       # Yeni kol MJCF -> URDF + mesh (kıskaç döndürme/açıklık sapmaları)
│   ├── check_urdf.py         # URDF doğrulama aracı
│   ├── check_robot_control.py # Eklem adım cevabı + kavrama doğrulaması
│   ├── view_multi_shape.py   # Görev sahnesi görüntüleyici
│   ├── teleop_maniskill_ps4.py # PS4 / gamepad / klavye ile canlı görev teleopu (kayıt yok)
│   ├── test_ps4.py           # Gamepad donanım test betiği
│   ├── record_vla_demos.py   # VLA teleop kaydı (telefon/PS4/klavye; queue/text/voice komut)
│   ├── generate_scripted_demos.py # Otomatik scripted VLA demoları (aynı şema)
│   ├── validate_dataset.py   # Veri kümesi doğrulama + kapsama + held-out sızıntı raporu
│   ├── compute_workspace.py  # Erişilebilir çalışma alanını FK/IK ile örnekler
│   ├── render_task_cameras.py# front/wrist kameralarını render edip yön tanımlarını doğrular
│   ├── make_heldout_split.py # Held-out komut listesini (bir kez) üretir
│   ├── make_act_dataset.py   # ACT için türev veri kümesi (5 slot -> 17 boyutlu one-hot)
│   ├── open_loop_check.py    # Açık çevrim (ezber) hata ölçümü
│   └── eval_policy.py        # Çok şekilli görevde kapalı çevrim değerlendirme (dil testi dahil)
├── src/mahut_kol/
│   ├── robots/               # MahutRobot (BaseAgent) + kinematics.py (URDF FK/IK)
│   ├── envs/                 # multi_shape_task.py, scripted_policy.py, shape_objects.py, prism_mesh.py
│   ├── commands/             # Komut şeması, İngilizce talimat, slotlar, dengeli kuyruk, held-out
│   ├── recording/            # writer.py (yazıcı, sidecar), session.py (döngü), ui.py (arayüz),
│   │                         # record_teleop.py / record_scripted.py (kayıt), validate.py, act_dataset.py
│   ├── evaluation.py         # Kapalı çevrim değerlendirme (scripts/eval_policy.py çağırır)
│   ├── teleop/               # ActionSource, PS4/klavye (teleop_ps4.py), telefon (phone_webxr.py eşleme, phone_server.py sunucu)
│   └── speech/               # Yerel Türkçe ses -> komut hattı
├── tests/                    # pytest
├── Yeni_model/               # Fusion MJCF + mesh kaynağı (git'e girmez)
├── requirements.txt, pyproject.toml
├── YAPILANLAR.md             # Kronolojik geliştirme günlüğü
└── PROJE_REHBERI.md          # Bu dosya (tek genel kaynak)
```

Kayıt betikleri veriyi doğrudan `LeRobotDataset` (lerobot 0.4.4, codebase v3.0) formatında yazar;
ara HDF5 adımı yoktur. Yeni kodu `datasets/` adlı klasöre koymayın (.gitignore yok sayar).

---

## 5. 🎮 Teleoperasyon ve Kontrol Seçenekleri

### A. PS4 Kolu (DualShock 4) / Klavye
DualShock 4 Linux'ta USB (`/dev/input/js0`) ya da Bluetooth (Share + PS basılı, "Wireless Controller")
ile tak-çalıştır tanınır; kol yoksa klavyeye düşülür. Test: `python3 scripts/test_ps4.py`,
canlı sürüş: `python3 scripts/teleop_maniskill_ps4.py [--seed 0]`, kayıt: §8 (`--device gamepad`).

* **Sol Analog:** Taban Dönüşü (`joint0` Yaw) ve Omuz Eğimi (`joint1` Pitch)
* **Sağ Analog:** Dirsek Eğimi (`joint2` Pitch) ve Bilek Eğimi (`joint3` Pitch)
* **D-Pad Sol/Sağ:** Bilek Yuvarlanma (`joint4` Roll)
* **R2 / L2 Tetikleri:** Kıskaç Kapat / Aç
* **Üçgen / Options:** Epizotu Sıfırla (Reset)
* **Klavye:** A/D (Taban), W/S (Omuz), Yukarı/Aşağı (Dirsek), Sol/Sağ (Bilek), Q/E (Roll), Boşluk (Kıskaç Kapat), F (Kıskaç Aç), R (Reset)

### B. Telefon (WebXR)
Telefonun 6-DoF pozu IK ile kola haritalanır; adım adım kurulum §8 "Telefonla teleop (WebXR)".

---

## 6. 🚀 Hızlı Başlangıç ve Komut Kılavuzu

### 1. Görev sahnesini izleme:
```bash
python scripts/view_multi_shape.py
```

### 2. Yeni kol URDF'ini MJCF'ten yeniden üretme:
```bash
python scripts/mjcf_to_urdf.py   # Yeni_model/arm_mujoco.xml -> assets/urdf + assets/meshes
```

### 3. Robot kontrolcüsü doğrulaması (adım cevabı + kavrama):
```bash
python scripts/check_robot_control.py
```

### 4. URDF Doğrulama ve Eklem Limitleri Testi:
```bash
python scripts/check_urdf.py
```

### 5. Eğitim İçin Gösterim (Demonstration) Toplama:
```bash
python scripts/collect_demos.py 50
```

### 6. Modal'da SmolVLA / ACT Eğitimi:
Ayrıntı: bölüm 9.
```bash
~/.venvs/lerobot/bin/python cloud/mahut_cloud.py train --config configs/training/smolvla.yaml --run <benzersiz_ad>
```

### 7. Eğitilen Modeli Simülasyonda Değerlendirme (çok şekilli görev):
```bash
~/.venvs/lerobot/bin/python scripts/eval_policy.py --checkpoint latest --episodes 45 --heldout-episodes 30 --language-scenes 10
# model olmadan boru hattı duman testi:
~/.venvs/lerobot/bin/python scripts/eval_policy.py --policy scripted --episodes 3 --heldout-episodes 2 --language-scenes 1
```

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
* **wrist:** kıskacın uç kısmına, üst tarafa (TCP çerçevesi) monte, parmak uçlarına bakar, 224×224.
Değerler `src/mahut_kol/envs/multi_shape_task.py` başındaki `CAMERA_*` sabitlerinde tek yerden değişir.

### Veri şeması (LeRobotDataset, lerobot 0.4.4, fps 20)
| Özellik | Tür | Şekil |
|---|---|---|
| `observation.images.front`, `observation.images.wrist` | video | 224×224×3 |
| `observation.state` | float32 | 6 (joint0-4, kıskaç servo açısı rad; `robot_state(qpos)`) |
| `action` | float32 | 6 (mutlak `pd_joint_pos` hedefi: joint0-4 + kıskaç 0..π rad) |
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
7. **DEBRİYAJ** düğmesini basılı tutup telefonu hareket ettirin. Pencerede kontrolün göre olduğu
   görüntü yeşil çerçeve ve etiketle işaretlidir; öteki görüntüde yönler terstir. Varsayılan
   kontrol **kıskaç (wrist) kamerasına göredir**: telefonu sağa götürmek → wrist görüntüsünde sağa, ileri itmek →
   görüntünün üstüne doğru, yukarı kaldırmak → yukarı (çerçeve her debriyaj basışında kıskacın o
   anki yönüne göre sabitlenir). Tek basışta uzun yana hareket ederseniz kol tabanda döner ve
   yönler kayar; kayma 30°'yi geçince pencerenin durum satırında **"YÖN KAYDI: debriyajı bırakıp
   tekrar basın"** uyarısı çıkar. Bırakıp yeniden basınca yönler görüntüye yeniden oturur. Eski front kamera eşlemesi için `--control-frame world`
   (ileri → robota doğru −x, sağa → +y). Telefonun burnunu aşağı
   eğmek kıskacı dikleştirir; direksiyon gibi çevirmek bilek roll'ü. Düğmeyi bırakınca kol durur
   (yeniden konumlanmak için bırakıp telefonu rahat pozisyona getirin). Bölüm ve süre sayacı
   (varsayılan 1200 adım = 60 s, `--max-steps`) debriyaja **ilk basıldığında** başlar; o ana dek
   pencerede "Telefon bekleniyor / debriyaja basın" yazar ve kare kaydedilmez. Süre dolarsa
   bölüm atılır ve beklemeden sıradaki sahne gelir. Terminalde saniyede bir `[Telefon] ...` tanı
   satırı paket hızını, debriyajı, telefon deltasını ve hedef/gerçek TCP'yi gösterir.
8. Pencerede hedef **kabul bölgesi** yarı saydam siyah daireyle (yarıçap 2,5 cm, beyaz nokta =
   tam hedef) iki kamerada da gösterilir; başarıda kenarı yeşile döner. İşaret sahneye eklenmez,
   yalnızca ekrana çizilir: kaydedilen görüntüler işaretsizdir (`--no-goal-marker` ile kapatılır).
   Kaydırıcı: %100 tam açık, %0 kapalı. Nesneyi kavrayıp hedefe koyun, kaydırıcıyı açarak bırakın.
9. Başarı algılanınca pencere yeşile döner: **Enter** kaydet, **R** at. Telefondaki "Sıfırla"
   düğmesi bölümü atıp yeniden başlatır.
10. Ters giden bir eksen varsa `src/mahut_kol/teleop/phone_webxr.py` içindeki `CONFIG["axis_signs"]`
    / `tilt_from` / `roll_from` işaretlerini değiştirin.

### Çalışma alanı ve kavrama
Yeni kolda bilek (`joint4` → TCP) 18.5 cm, kol + önkoldan uzun; ±90° limitlerle masada dikey
kavrama yok, kavrama düşeyden 40–55° eğik ve parmaklar yatay (`joint4 = 0`). Nesne/hedef alanı
x ∈ [0.10, 0.22], y ∈ [−0.14, 0.14] m (`scripts/compute_workspace.py`, spawn erişilebilirliği
`--spawn-check`). Kinematik `src/mahut_kol/robots/kinematics.py`: URDF'ten saf numpy
FK (SAPIEN TCP pozuyla birebir), eğim kısıtlı DLS IK; telefon teleopu ve scripted demolar aynı
çözücüyü kullanır.

---

## 9. ☁️ Modal'da VLA Eğitimi (SmolVLA ve ACT)

Kod: `cloud/` (`modal_app.py` imaj + uzak fonksiyonlar, `train_entry.py` koşu/devam mantığı,
`mahut_cloud.py` komut satırı, `modal_yardim.py` ortak Modal yardımcıları). Koşu ayarları tek yerde: `configs/training/smolvla.yaml`, `act.yaml`
(politika, adım, batch, kaydetme sıklığı, GPU, timeout, veri adı). Wandb varsayılan kapalı.

### Kurulum (bir kez)
`~/.venvs/lerobot` içinde `modal` kurulu (1.6.3). Kimlik doğrulamayı **siz** yapın (anahtar hiçbir dosyaya yazılmaz):
```bash
~/.venvs/lerobot/bin/modal setup
~/.venvs/lerobot/bin/python cloud/mahut_cloud.py check     # profil + Volume'lar
```
Volume'lar: `mahut-datasets` (sürümlü veri kümeleri) ve `mahut-runs` (`runs/<koşu>/`, HF model önbelleği `hf_cache/`).

### Veri yükleme (sürümlü; var olan ad ASLA ezilmez)
```bash
# SmolVLA: ham veri kümesi
python cloud/mahut_cloud.py upload --src ~/vla_data/mahut_smoke_v1 --name mahut_smoke_v1_r1
# ACT (dilsiz kıyas): önce türev veri kümesi, sonra yükle
python scripts/make_act_dataset.py --src ~/vla_data/mahut_smoke_v1 --dst ~/vla_data/mahut_smoke_v1_act
python cloud/mahut_cloud.py upload --src ~/vla_data/mahut_smoke_v1_act --name mahut_smoke_v1_r1_act
```
Veri güncellenirse yeni ad kullanın (`..._r2`). Yalnızca `meta/`, `data/`, `videos/` yüklenir.
**Gerçek teleop verisi (`~/vla_data/mahut_multishape`) bu hat denemelerinde yüklenmedi**; yüklemeden önce karar sizin.

### Eğitim, devam, indirme
```bash
python cloud/mahut_cloud.py train --config configs/training/smolvla.yaml --run smolvla_x_01 --steps 300   # YENİ koşu
python cloud/mahut_cloud.py train --config configs/training/smolvla.yaml --run smolvla_x_01 --steps 600 --resume  # kaldığı yerden
python cloud/mahut_cloud.py ls [--run smolvla_x_01]
python cloud/mahut_cloud.py download --run smolvla_x_01              # ~/vla_runs/smolvla_x_01/checkpoints/last/pretrained_model
python cloud/mahut_cloud.py download --run smolvla_x_01 --with-training-state   # devam/taşıma için optimizer durumu dahil
modal billing report --for today --show-resources                    # GERÇEK harcama (on_tahmin_usd yalnızca ön tahmin)
```
* Koşu adı benzersizdir: var olan koşu `--resume` olmadan HATA verir; silme/ezme yoktur.
* `--detach` ile başlar (terminal kapansa da sürer). Her fonksiyonda `timeout_min` (varsayılan 30) ve en fazla 30 dk sınırı
  (`--max-timeout-min` ile bilerek yükseltilir). `max_retries: 2`: zaman aşımı/preemption sonrası Modal aynı girdiyle
  yeniden başlatır, `train_entry.py` son checkpoint'ten sürdürür (gerçek zaman aşımıyla doğrulandı).
* Checkpoint `save_freq` adımda bir yazılır, Volume'a 45 sn'de bir ve çıkışta `commit` edilir.
* GPU: `--gpu L4`; kapasite yoksa yedekli liste `--gpu "L4,A10"` (deneme sırasında L4 kapasitesi 13 dk beklendi).
* Video okuma: `pyav` açıkça seçili (`video_backend: pyav`).
* SmolVLA: `front`→`camera1`, `wrist`→`camera2`, `empty_cameras: 1` (smolvla_base 3 kamera ile eğitildi).
* ACT: komut 5 slottan 17 boyutlu one-hot (`observation.environment_state`); türev veri kümesi gerekir.

### Yerel değerlendirme
```bash
python scripts/eval_policy.py --checkpoint ~/vla_runs/<koşu>/checkpoints/last/pretrained_model --episodes 5 --heldout-episodes 1 --language-scenes 1
python scripts/open_loop_check.py --checkpoint ... --root ~/vla_data/mahut_smoke_v1[_act] --episodes 0-9   # ezber hatası
```
Rapor `policy_stats` içinde eylem aralığı, çıkarım gecikmesi ve GPU bellek tepe noktasını verir.

### Imaj / yerel venv farkları
Aynı: Python 3.10 (Modal 3.10.17, yerel 3.10.12), torch 2.6.0, torchvision 0.21.0, lerobot 0.4.4, transformers 4.57.6,
numpy 2.2.6, av 15.1.0, datasets 4.8.5, diffusers 0.35.2, accelerate 1.15.0, safetensors 0.8.0, huggingface-hub 0.35.3.
Farklı: Modal imajında ManiSkill/SAPIEN ve `mahut_kol` YOK (yalnızca eğitim); yerel venv sistem paketlerini de görür;
ffmpeg Debian 5.1.9 (apt); yalnızca `opencv-python-headless`; torchcodec kurulu ama kullanılmıyor.

### Ölçüm tablosu (kısa denemeler, 250 scripted bölüm, batch 16)
Ayrıntılar `YAPILANLAR.md` §16.

---

## 10. 🔁 Eğitimi Kişiler Arasında Devretme (Data_Model + `cloud/devir.py`)

Arkadaşların adım adım izleyeceği kılavuz: [docs/EGITIM_DEVIR_KILAVUZU.md](docs/EGITIM_DEVIR_KILAVUZU.md).
Herkes **kendi** Modal hesabını kendi bilgisayarında kullanır (`modal setup`); token paylaşımı, profil değiştirme ya da
çoklu hesap özelliği yoktur. Model Drive'da durur, dosyalar tarayıcıdan elle yüklenip indirilir (rclone/Drive API yok).
Kod Drive'a konmaz: herkes aynı git commit'inden çalışır.

```text
Data_Model/                         # git'e girmez (yalnızca README.md izlenir)
├── sabit/sabit_v1.tar              # veri kümesi (+ACT türevi, videolar tek kopya); bir kez indirilir, değişmez
└── iterasyonlar/<hat>/iter_0002_step_010000_ali.tar   # pretrained_model + training_state + train.log + attempts.jsonl + manifest.json
```
Manifest: format sürümü, hat, iterasyon, adım, kişi (kısa ad), **parent**, politika, git commit, `configs/training` içeriği,
sabit sürümü, paket sürümleri, GPU, tur süresi, her dosyanın sha256'sı. Tar'lar sıkıştırmasız ve yalnızca
`.safetensors/.json/.jsonl/.log` düz dosyalarından oluşur; pickle, mutlak yol, `..`, sembolik/sert bağ reddedilir.

| Komut | İş |
|---|---|
| `devir.py durum --hat H` | sabit, son iterasyon, profil (kimliksiz), bu ayki harcama, sıradaki no, azami adım |
| `devir.py sabit-olustur --surum 1 --veri V [--act-veri A]` | `Data_Model/sabit/sabit_v1.tar` |
| `devir.py kur` | sabit tar'ı doğrular, kendi Volume'una yükler (özet aynıysa atlar) |
| `devir.py basla --hat H --config configs/training/act.yaml --kisi K --adim N` | ebeveynsiz ilk tur |
| `devir.py devam --hat H --kisi K --adim N [--sec dosya]` | son tar'ı doğrular, Volume'a geri yükler, hedef = mevcut adım + N |
| `devir.py paketle --hat H --kisi K` | tam checkpoint'i indirir, tar'ı üretir ve doğrular |
| `devir.py temizle --hat H [--onayla]` | paketlenmiş eski checkpoint'leri KENDİ Volume'undan siler (en sonuna dokunmaz) |

**Geri yükleme (restore):** koşu adı = hat adı ve veri adları sabit pakette sabit olduğundan `train_config.json`
içindeki mutlak yollar (`/runs/runs/<hat>/train`, `/tmp/ds/<veri>`) her hesapta aynıdır; yollar yeniden yazılmaz.
Checkpoint `/runs/<hat>/train/checkpoints/<adım>/` altına konur, `launch.json` yoksa yazılır, `checkpoints/last`
bağı gerekmez. Aynı adım Volume'da varsa özet aynıysa atlanır, farklıysa hata verilir.

**`devam` öncesi denetimler (biri başarısızsa eğitim başlamaz):** tar okunur + tüm sha256 tutar + güvenli üyeler; git
temiz (Data_Model/ hariç) ve `cloud/`, `configs/training/`, `src/mahut_kol/commands/` manifest commit'iyle aynı; Modal
imajındaki paket sürümleri manifest ile aynı; sabit sürümü ve hat eşleşir; çatallanma yok (`--sec` ile çözülür);
`--adim` kredi sınırı ve zaman aşımı tavanı içinde.

**Kredi:** `modal billing summary --json` ile bu ayki gerçek harcama (kredi + faturalanan) okunur; 30 USD'den 5 USD
güvenlik payı düşülür. Politika bazında maliyet `configs/training/*.yaml` → `maliyet` bölümündedir (SmolVLA ≈ 0,20,
ACT ≈ 0,05 USD/1000 adım + sabit pay). `save_freq`: ACT 5000 (~9 dk), SmolVLA 1500 (~13 dk).
`timeout_min` `--adim`'den hesaplanır, `--max-timeout-min` (varsayılan 90) hâlâ geçerlidir.
