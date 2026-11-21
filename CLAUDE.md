# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje

MahutKol: özel 5-DOF robot kol + çift parmak paralel tutucu. Depo dört ana hattan oluşur:

- **`Main_project/`**: ana hat. ManiSkill 3 (SAPIEN/PhysX) simülasyonu, teleoperasyonla demo toplama, LeRobot formatı, VLA eğitimi (ACT/SmolVLA) ve robotumuza göre ince ayar, kapalı çevrim değerlendirme. Python paketi `mahut_kol` burada.
- **`Real-Sim/`**: gerçek kol ↔ simülasyon/bilgisayar köprüsü (ESP32-S3 MicroPython firmware, `lerobot_robot_mahut` sürücüsü).
- **`Paper/`**: Türkçe dil talimatlarıyla LIBERO üzerinde sonuçlar.
- **`Side_project/`**: yalnızca MuJoCo teleoperasyon (SO-100); ana pakete bağlı değil.

Kökte ortak olanlar: `Fikirler/` (PM notları/promptlar), `graphify-out/`. Dokümantasyon ve yorumlar çoğunlukla Türkçe; yeni içerikte aynı dili koru. Ana hattın kılavuzları: `Main_project/PROJE_REHBERI.md`, `Main_project/docs/`, ilerleme günlüğü `Main_project/YAPILANLAR.md`, görev spesifikasyonu `Main_project/PROJECT.md`.

## Komutlar (Main_project)

Aşağıdaki tüm komutlar ve yollar `Main_project/` içinden çalıştırılır (`cd Main_project`). Bu bölümün geri kalanındaki göreli yollar (`src/`, `scripts/`, `cloud/`, `configs/`, `assets/`, `docs/`, `Data_Model/`, `Yeni_model/`) `Main_project/` altındadır.

```bash
pip install -r requirements.txt && pip install -e .   # Python >=3.10, numpy<2
pytest tests/                                          # tüm testler (tests/ pytest, scripts/ ise bağımsız doğrulama betikleri)
pytest tests/test_command_parser.py -k <isim>          # tek test
pytest tests/test_multi_shape_task.py tests/test_shape_objects.py tests/test_prism.py   # 9 nesneli ortam doğrulaması
python3 scripts/view_multi_shape.py                    # sahneyi görsel olarak izle
python3 scripts/test_voice_command.py --mode text --input-text "Kırmızı küpü al yeşil silindirin sağına koy"
python3 scripts/test_voice_command.py --mode push_to_talk|vad --model base   # mikrofon gerekir
ruff check src tests ; black src tests                 # dev bağımlılıkları (pyproject [dev])
```

VLA veri hattı (`~/.venvs/lerobot/bin/python` ile; ayrıntı `PROJE_REHBERI.md` §8, `docs/VLA_GOREV_TANIMLARI.md`):

```bash
python scripts/record_vla_demos.py --device phone|gamepad --command-source queue|text|voice   # teleop kaydı
python scripts/generate_scripted_demos.py --num-episodes N    # scripted demolar, aynı veri kümesine ekler
python scripts/validate_dataset.py --video-check sample       # şema/kapsama/held-out sızıntı raporu
python scripts/eval_policy.py --checkpoint <yol>|latest | --policy scripted   # kapalı çevrim değerlendirme (ACT one-hot / SmolVLA otomatik)
python scripts/open_loop_check.py --checkpoint <yol> --root <veri>   # açık çevrim ezber hatası
python scripts/compute_workspace.py ; python scripts/render_task_cameras.py   # çalışma alanı / kamera doğrulama
```

Modal eğitim (`cloud/`, ayrıntı `PROJE_REHBERI.md` §9; Modal kimlik doğrulamasını kullanıcı yapar):

```bash
python cloud/mahut_cloud.py upload --src <veri_kökü> --name <sürümlü_ad>      # ezmez, var olan ad hata
python scripts/make_act_dataset.py --src <kök> --dst <kök>_act                 # ACT için 17 boyutlu one-hot türev küme
python cloud/mahut_cloud.py train --config configs/training/{smolvla,act}.yaml --run <benzersiz_ad> [--resume] [--gpu "L4,A10"]
python cloud/mahut_cloud.py download --run <ad> ; python cloud/mahut_cloud.py ls
```

Eğitim devri (`cloud/devir.py`, ayrıntı `PROJE_REHBERI.md` §10, arkadaş kılavuzu `docs/EGITIM_DEVIR_KILAVUZU.md`; yalnızca modal+pyyaml gerekir, herkes KENDİ Modal hesabını kullanır):

```bash
python cloud/devir.py sabit-olustur --surum 1 --veri <kök> [--act-veri <kök>_act]   # Data_Model/sabit/sabit_v1.tar
python cloud/devir.py kur ; python cloud/devir.py basla|devam --hat H --kisi K --adim N ; python cloud/devir.py paketle --hat H --kisi K
```

Varsayılan veri kökü `~/vla_data/mahut_multishape` (repo_id `local/mahut_multishape`); veri her zaman eklenir, asla silinmez. Robot doğrulama: `scripts/check_urdf.py` (kütle/çarpışma/kendi kendine çarpışma), `scripts/check_robot_control.py` (adım cevabı + kavrama), `scripts/compute_workspace.py --spawn-check 300`.

## Mimari (Main_project)

Paket `Main_project/src/mahut_kol/` (setuptools, `where = src`). Akış: `robots` → `envs` → `commands` → `teleop` / scripted → `recording` (doğrudan LeRobot v3 formatı) → eğitim (Modal) → değerlendirme; yan modül `speech`. Eski `datasets/`, `train/`, `eval/` stub paketleri silindi; `.gitignore` içindeki `datasets/` kuralı yüzünden yeni kodu `datasets/` adlı klasöre koyma.

- **`robots/mahut_robot.py`**: `@register_agent()` ile `MahutRobot` (ManiSkill `BaseAgent`). Yeni kol (2026-11): URDF `assets/urdf/mahut_kol.urdf`, `scripts/mjcf_to_urdf.py` ile `Yeni_model/arm_mujoco.xml`'den (git'te yok) üretilir, elle düzenleme. Eklemler: `joint0..4` (FT5330M, ±90°) + `finger_left/right` (0 = açık, 0.022 m = kapalı). Eylem/state 6 boyut: 5 kol + kıskaç servo açısı (rad, 0 = açık, π = kapalı); robot qpos 7, dönüşüm `robot_state()`. Kıskaç bilinçli olarak CAD'den sapar: roll servosuna 90° döndürülmüş montaj (`joint4 = 0`'da parmaklar yatay) ve 46 mm açıklık (CAD 38 mm). **`robots/kinematics.py`**: URDF'ten saf numpy FK (SAPIEN TCP ile birebir), eğim kısıtlı DLS IK, `solve_grasp_ik`, `wrap_roll`, parmak-masa kontrolü. Bilek 18.5 cm: masada dikey kavrama yok, kavrama düşeyden 40–55° eğik.
- **`envs/`**: görev ortamı. `multi_shape_task.py` içindeki `MultiShapeRelationalTask` 9 nesneli (3 şekil × 3 renk) ilişkisel yerleştirme görevidir; nesneleri `shape_objects.build_manipulation_objects()` üretir, üçgen prizma meshi `assets/objects/triangular_prism.obj` (`envs/prism_mesh.py`, betik `scripts/generate_prism_mesh.py`). Aktif olmayan nesneler z = -10.0'a "park" edilir (PhysX'in zemin fırlatmasını önlemek için); çakışmasız spawn rejection sampling ile yapılır; hedef pozlar FRONT/BACK/LEFT/RIGHT/TOP ilişkilerinden hesaplanır (Δ = 0.06 m; front kamera operatör tarafında, görüntü sağı = +y = RIGHT). Kurallar: referans prizma + TOP geçersiz (336 geçerli komut), hedef boş ve çalışma alanında, başarı = yerleşti + bırakıldı + durağan; kameralar `front` + `wrist` (224×224, sabitler dosya başında), 600 adım. Varsayılan robot `mahut_kol`. `envs/scripted_policy.py`: otomatik gösterim politikası.
- **`commands/`**: `Command` (parser JSON ↔ reset options), İngilizce talimat şablonları, 5 slot, `BalancedCommandQueue` (devam ettirilebilir), held-out (`configs/commands/heldout_commands.json`), komut kaynakları queue/text/voice.
- **`recording/`**: `VLADatasetWriter` (LeRobot oluştur/aç-ekle, şema koruması, sidecar `mahut_episodes.jsonl`), kayıt döngüsü (`session.py`), pygame/terminal arayüzü ve hedef işareti (`ui.py`), teleop/scripted kayıt (`record_teleop.py`, `record_scripted.py`), doğrulama (`validate.py`), ACT türev kümesi (`act_dataset.py`). Paket adı bilinçli olarak `datasets` değil.
- **`speech/`**: tamamen yerel Türkçe sesli komut hattı: `audio_recorder` (sounddevice, 16 kHz mono, push-to-talk/VAD) → `stt_engine` (faster-whisper; CUDA'da float16, CPU'da int8) → `command_parser` (kural tabanlı, Türkçe çekim/bulunma ekleri, `MultiShapeRelationalTask` uzamsal ilişkilerine eşlenir) → `voice_commander` (hepsini birleştiren `VoiceCommander`).
- **`evaluation.py`**: kapalı çevrim değerlendirme (LeRobot politika sarmalayıcısı, scripted kâhin, dil varyantları).
- **`scripts/` yalnızca giriş noktasıdır**: mantık `src/mahut_kol/` içinde; testlerin ihtiyaç duyduğu kod betikte durmaz. `record_vla_demos.py`, `generate_scripted_demos.py`, `validate_dataset.py`, `make_act_dataset.py`, `eval_policy.py`, `generate_prism_mesh.py` ilgili modülün `main()`'ini çağıran sarmalayıcılardır; bağımsız araçlar (`check_*`, `compute_workspace`, `render_task_cameras`, `mjcf_to_urdf`, `view_multi_shape`, `open_loop_check`, `test_*`) betikte kalır.
- **`teleop/`**: tek teleop yeri. `teleop_ps4.py` (`MahutPS4Controller`, gamepad/klavye okuyucu), `action_source.py` cihazdan bağımsız `ActionSource` (6 boyutlu `pd_joint_pos`), `phone_webxr.py` telefon → eklem eşlemesi + `PhoneActionSource`, `phone_server.py` WebSocket sunucusu + protokol, `web/index.html` sayfa (Side_project'ten uyarlanmış kopya). Canlı sürüş/test betikleri `scripts/teleop_maniskill_ps4.py`, `scripts/test_ps4.py`. HDF5 kaydı yok; kayıt yalnızca LeRobot (`record_vla_demos.py`).
- **`cloud/`**: Modal eğitim hattı (`modal_app.py` imaj+Volume+uzak fonksiyon, `train_entry.py` koşu/devam mantığı, `mahut_cloud.py` CLI, `modal_yardim.py` iki CLI'nin ortak Modal yardımcıları). Koşu ayarları `configs/training/*.yaml`. ACT dil almaz: `commands/onehot.py` 5 slotu 17 boyutlu `observation.environment_state`'e çevirir (eğitim türev veri kümesiyle, `eval_policy.py` aynı fonksiyonla).
- **`cloud/devir.py` + `devir_lib.py`**: eğitimin kişiler arası devri (güvenli tar + manifest/sha256, restore, çatallanma, kredi koruması). `Data_Model/` (Drive ile aynı düzen) git'e girmez, yalnızca `Data_Model/README.md` izlenir. `MAHUT_VOLUME_SUFFIX` yalnızca teste özeldir.
- **`configs/`**: `robot/`, `task/`, `training/` YAML'ları (`training/*.yaml` içinde `maliyet` kredi modeli); `commands/heldout_commands.json`.
- **`assets/`**: yeni kolun üretilmiş STL mesh'leri ve URDF'i, nesne meshleri. **`Side_project/`**: SO-100 MuJoCo/WebXR/webcam teleop denemeleri, ana pakete bağlı değil.

## Notlar

- **`Side_project/` içindeki dosyalara DOKUNMA** (özellikle MuJoCo `webxr_teleop.py`, `ik_solver.py`, `index.html`). Gerekirse içeriği okuyup yeni dosyayı ana pakette yaz; orijinal olduğu gibi kalsın.
- LeRobot ve VLA eğitimi için ayrı venv: `~/.venvs/lerobot` (lerobot 0.4.4 + smolvla, sistem paketlerini görür; numpy 2.2.6 ile sorunsuz). Eğitim ileride Modal'da yapılacak, yerel GPU yalnızca simülasyon/inference. Yerelde torchcodec FFmpeg kütüphanelerini bulamıyor; LeRobot videolarını okurken `video_backend="pyav"` kullan.
- Testlerde tek süreçte birden fazla ortam açılıyorsa `render_backend="cpu"` verme (SAPIEN ikinci renderer'ı açamaz); varsayılanı kullan.

- **Kod tabanını keşfetmeden önce `graphify-out/GRAPH_REPORT.md`'yi oku**: graphify skill'inin ürettiği bilgi grafı özeti (topluluklar, merkez düğümler, dosya ilişkileri). Mimari/ilişki sorularında önce graphify skill'ini kullan; geniş dosya taraması yerine buradan ilgili bölgeyi bulup sadece o dosyaları aç. `graph.json`/`graph.html` büyük dosyalardır, doğrudan okuma. Grafik eskiyebilir (son güncelleme commit'i `docs(graph): ...`); büyük değişikliklerden sonra graphify ile yenile.
- `graphify-out/` ve `.agents/` otomatik üretilmiş çıktılardır; elle düzenleme.
- Git commit mesajları conventional commits stilinde (`feat(speech): ...`, `fix(...)`, `docs: ...`).
