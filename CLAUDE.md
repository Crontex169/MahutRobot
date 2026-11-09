# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proje

MahutKol: özel 5-DOF robot kol + çift parmak paralel tutucu için ManiSkill 3 (SAPIEN/PhysX) simülasyonu, teleoperasyonla demo toplama, LeRobot formatına çevirme, politika eğitimi (ACT/SmolVLA/Diffusion) ve kapalı çevrim değerlendirme. Dokümantasyon ve yorumlar çoğunlukla Türkçe; yeni içerikte aynı dili koru. Ayrıntılı kılavuzlar: `PROJE_REHBERI.md`, `docs/`, ilerleme günlüğü `YAPILANLAR.md`, çok-şekilli görev spesifikasyonu `PROJECT.md`.

## Komutlar

```bash
pip install -r requirements.txt && pip install -e .   # Python >=3.10, numpy<2
pytest tests/                                          # tüm testler (tests/ pytest, scripts/ ise bağımsız doğrulama betikleri)
pytest tests/test_command_parser.py -k <isim>          # tek test
python3 scripts/test_shapes_env.py --episodes 5        # 9 nesneli ortam doğrulama paketi (--render, --save-image)
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

Varsayılan veri kökü `~/vla_data/mahut_multishape` (repo_id `local/mahut_multishape`); veri her zaman eklenir, asla silinmez. Diğer: `scripts/collect_demos.py` (eski PickCube), URDF doğrulama `scripts/check_urdf.py`.

## Mimari

Paket `src/mahut_kol/` (setuptools, `where = src`). Akış: `robots` → `envs` → `commands` → `teleop` / scripted → `recording` (doğrudan LeRobot v3 formatı) → eğitim (Modal) → değerlendirme; yan modül `speech`. Eski `datasets/`, `train/`, `eval/` stub paketleri silindi; `.gitignore` içindeki `datasets/` kuralı yüzünden yeni kodu `datasets/` adlı klasöre koyma.

- **`robots/mahut_robot.py`**: `@register_agent()` ile `MahutRobot` (ManiSkill `BaseAgent`). URDF'i `assets/urdf/mahut_kol.urdf`'ten (MJCF alternatifi `assets/mahut_kol_maniskill.xml`) göreli yolla yükler; kontrolcüler `robots/controllers.py`. Eklemler: `joint0..4` + `finger_left/right` (parmak 0 = açık, 0.042 = kapalı). **`robots/kinematics.py`**: URDF'ten saf numpy FK (SAPIEN TCP ile birebir), eğim kısıtlı DLS IK, `solve_grasp_ik`, parmak-masa kontrolü. MahutKol tam aşağı bakarak çalışma alanına erişemez; kavrama düşeyden 45–55° eğik.
- **`envs/`**: görev ortamları (`pick_cube_task`, `push_cube_task`, `minimal_task`). `multi_shape_task.py` içindeki `MultiShapeRelationalTask` 9 nesneli (3 şekil × 3 renk) ilişkisel yerleştirme görevidir; nesneleri `shape_objects.build_manipulation_objects()` üretir, üçgen prizma meshi `assets/objects/triangular_prism.obj` (`scripts/generate_prism_mesh.py` ile üretilir). Aktif olmayan nesneler z = -10.0'a "park" edilir (PhysX'in zemin fırlatmasını önlemek için); çakışmasız spawn rejection sampling ile yapılır; hedef pozlar FRONT/BACK/LEFT/RIGHT/TOP ilişkilerinden hesaplanır (Δ = 0.06 m; front kamera operatör tarafında, görüntü sağı = +y = RIGHT). Kurallar: referans prizma + TOP geçersiz (336 geçerli komut), hedef boş ve çalışma alanında, başarı = yerleşti + bırakıldı + durağan; kameralar `front` + `wrist` (224×224, sabitler dosya başında), 600 adım. Varsayılan robot `mahut_kol`. `envs/scripted_policy.py`: otomatik gösterim politikası.
- **`commands/`**: `Command` (parser JSON ↔ reset options), İngilizce talimat şablonları, 5 slot, `BalancedCommandQueue` (devam ettirilebilir), held-out (`configs/commands/heldout_commands.json`), komut kaynakları queue/text/voice.
- **`recording/`**: `VLADatasetWriter` (LeRobot oluştur/aç-ekle, şema koruması, sidecar `mahut_episodes.jsonl`), kayıt döngüsü ve pygame/terminal arayüzü. Paket adı bilinçli olarak `datasets` değil.
- **`speech/`**: tamamen yerel Türkçe sesli komut hattı: `audio_recorder` (sounddevice, 16 kHz mono, push-to-talk/VAD) → `stt_engine` (faster-whisper; CUDA'da float16, CPU'da int8) → `command_parser` (kural tabanlı, Türkçe çekim/bulunma ekleri, `MultiShapeRelationalTask` uzamsal ilişkilerine eşlenir) → `voice_commander` (hepsini birleştiren `VoiceCommander`).
- **`teleop/`** ve kökteki **`Teleoperasyonlar/`**: PS4/klavye kontrolcüleri ve demo kaydı; iki yerde benzer kod var (`utils/teleop_ps4.py` da dahil), değişiklik yaparken hangisinin kullanıldığını kontrol et. `teleop/action_source.py` cihazdan bağımsız `ActionSource` (7 boyutlu `pd_joint_pos`); `teleop/phone_webxr.py` + `teleop/web/index.html` telefon WebXR teleopu (Side_project'ten uyarlanmış kopya).
- **`cloud/`**: Modal eğitim hattı (`modal_app.py` imaj+Volume+uzak fonksiyon, `train_entry.py` koşu/devam mantığı, `mahut_cloud.py` CLI). Koşu ayarları `configs/training/*.yaml`. ACT dil almaz: `commands/onehot.py` 5 slotu 17 boyutlu `observation.environment_state`'e çevirir (eğitim türev veri kümesiyle, `eval_policy.py` aynı fonksiyonla).
- **`cloud/devir.py` + `devir_lib.py`**: eğitimin kişiler arası devri (güvenli tar + manifest/sha256, restore, çatallanma, kredi koruması). `Data_Model/` (Drive ile aynı düzen) git'e girmez, yalnızca `Data_Model/README.md` izlenir. `MAHUT_VOLUME_SUFFIX` yalnızca teste özeldir.
- **`configs/`**: `robot/`, `task/`, `training/` YAML'ları (`training/*.yaml` içinde `maliyet` kredi modeli); `commands/heldout_commands.json`.
- **`assets/`**: robot STL mesh'leri, URDF/MJCF ve nesne meshleri. **`Side_project/`**: SO-100 MuJoCo/WebXR/webcam teleop denemeleri, ana pakete bağlı değil.

## Notlar

- **`Side_project/` içindeki dosyalara DOKUNMA** (özellikle MuJoCo `webxr_teleop.py`, `ik_solver.py`, `index.html`). Gerekirse içeriği okuyup yeni dosyayı ana pakette yaz; orijinal olduğu gibi kalsın.
- LeRobot ve VLA eğitimi için ayrı venv: `~/.venvs/lerobot` (lerobot 0.4.4 + smolvla, sistem paketlerini görür; numpy 2.2.6 ile sorunsuz). Eğitim ileride Modal'da yapılacak, yerel GPU yalnızca simülasyon/inference. Yerelde torchcodec FFmpeg kütüphanelerini bulamıyor; LeRobot videolarını okurken `video_backend="pyav"` kullan.
- Testlerde tek süreçte birden fazla ortam açılıyorsa `render_backend="cpu"` verme (SAPIEN ikinci renderer'ı açamaz); varsayılanı kullan.

- **Kod tabanını keşfetmeden önce `graphify-out/GRAPH_REPORT.md`'yi oku**: graphify skill'inin ürettiği bilgi grafı özeti (topluluklar, merkez düğümler, dosya ilişkileri). Mimari/ilişki sorularında önce graphify skill'ini kullan; geniş dosya taraması yerine buradan ilgili bölgeyi bulup sadece o dosyaları aç. `graph.json`/`graph.html` büyük dosyalardır, doğrudan okuma. Grafik eskiyebilir (son güncelleme commit'i `docs(graph): ...`); büyük değişikliklerden sonra graphify ile yenile.
- `graphify-out/` ve `.agents/` otomatik üretilmiş çıktılardır; elle düzenleme.
- Git commit mesajları conventional commits stilinde (`feat(speech): ...`, `fix(...)`, `docs: ...`).
