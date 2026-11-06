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

Pipeline betikleri: `scripts/collect_demos.py`, `scripts/train_smolvla.sh`, `scripts/eval_policy.py`; URDF doğrulama `scripts/check_urdf.py`.

## Mimari

Paket `src/mahut_kol/` (setuptools, `where = src`). Akış: `robots` → `envs` → `teleop` → `datasets` → `train` → `eval`; yan modül `speech`.

- **`robots/mahut_robot.py`**: `@register_agent()` ile `MahutRobot` (ManiSkill `BaseAgent`). URDF'i `assets/urdf/mahut_kol.urdf`'ten (MJCF alternatifi `assets/mahut_kol_maniskill.xml`) göreli yolla yükler; kontrolcüler `robots/controllers.py`. Eklemler: `joint0..4` + `finger_left/right`.
- **`envs/`**: görev ortamları (`pick_cube_task`, `push_cube_task`, `minimal_task`). `multi_shape_task.py` içindeki `MultiShapeRelationalTask` 9 nesneli (3 şekil × 3 renk) ilişkisel yerleştirme görevidir; nesneleri `shape_objects.build_manipulation_objects()` üretir, üçgen prizma meshi `assets/objects/triangular_prism.obj` (`scripts/generate_prism_mesh.py` ile üretilir). Aktif olmayan nesneler z = -10.0'a "park" edilir (PhysX'in zemin fırlatmasını önlemek için); çakışmasız spawn rejection sampling ile yapılır; hedef pozlar FRONT/BACK/LEFT/RIGHT/TOP ilişkilerinden hesaplanır. Varsayılan robot `so100`, `mahut_kol` ile uyumlu.
- **`speech/`**: tamamen yerel Türkçe sesli komut hattı: `audio_recorder` (sounddevice, 16 kHz mono, push-to-talk/VAD) → `stt_engine` (faster-whisper; CUDA'da float16, CPU'da int8) → `command_parser` (kural tabanlı, Türkçe çekim/bulunma ekleri, `MultiShapeRelationalTask` uzamsal ilişkilerine eşlenir) → `voice_commander` (hepsini birleştiren `VoiceCommander`).
- **`teleop/`** ve kökteki **`Teleoperasyonlar/`**: PS4/klavye kontrolcüleri ve demo kaydı; iki yerde benzer kod var (`utils/teleop_ps4.py` da dahil), değişiklik yaparken hangisinin kullanıldığını kontrol et.
- **`configs/`**: `robot/`, `task/`, `train/` YAML'ları (LeRobot ACT eğitimi vb.).
- **`assets/`**: robot STL mesh'leri, URDF/MJCF ve nesne meshleri. **`Side_project/`**: SO-100 MuJoCo/WebXR/webcam teleop denemeleri, ana pakete bağlı değil.

## Notlar

- **Kod tabanını keşfetmeden önce `graphify-out/GRAPH_REPORT.md`'yi oku**: graphify skill'inin ürettiği bilgi grafı özeti (topluluklar, merkez düğümler, dosya ilişkileri). Mimari/ilişki sorularında önce graphify skill'ini kullan; geniş dosya taraması yerine buradan ilgili bölgeyi bulup sadece o dosyaları aç. `graph.json`/`graph.html` büyük dosyalardır, doğrudan okuma. Grafik eskiyebilir (son güncelleme commit'i `docs(graph): ...`); büyük değişikliklerden sonra graphify ile yenile.
- `graphify-out/` ve `.agents/` otomatik üretilmiş çıktılardır; elle düzenleme.
- Git commit mesajları conventional commits stilinde (`feat(speech): ...`, `fix(...)`, `docs: ...`).
