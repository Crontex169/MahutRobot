# MahutKol: ManiSkill & LeRobot Tabanlı Robot Öğrenme Şablonu

Bu proje, özel tasarım bir robotik kol için **ManiSkill** simülasyon ortamında veri toplama, **LeRobot** (Hugging Face) ile politika eğitimi (Imitation Learning / ACT / Diffusion Policy) ve eğitilen modelin simülasyonda kapalı çevrim (closed-loop) test edilmesi için hazırlanmış modüler bir proje şablonudur.

---

## 🚀 İş Akışı Mimarisi (Pipeline)

```text
[CAD Tasarımı (STP/STL)]
       ↓ (CoACD + URDF Exporter)
[URDF + Mesh Dosyaları (assets/)]
       ↓
[ManiSkill Robot & Görev Entegrasyonu (src/mahut_kol/robots & envs)]
       ↓
[Teleoperasyon ile Veri Toplama (src/mahut_kol/teleop)]
       ↓
[ManiSkill ➔ LeRobot Formatına Dönüştürme (src/mahut_kol/datasets)]
       ↓
[LeRobot ile Politika Eğitimi (src/mahut_kol/train)]
       ↓
[Simülasyonda Kapalı Çevrim Değerlendirme (src/mahut_kol/eval)]
```

---

## 📁 Proje Klasör Yapısı

*   `assets/`: Robotun URDF, görsel (visual) ve çarpışma (collision) mesh dosyaları.
*   `docs/`: ManiSkill 3 ve LeRobot kütüphanelerinin resmi kılavuzları ve mimari referansları.
*   `configs/`: Robot, görev ortamı ve eğitim hiperparametre konfigürasyonları (YAML).
*   `Teleoperasyonlar/`: PS4 / Gamepad ve klavye teleoperasyon kontrolcüleri, test betikleri ve veri toplama araçları.
*   `src/mahut_kol/`:
    *   `robots/`: ManiSkill robot tanımları, aktüatör ve kontrolcü yapılandırmaları.
    *   `envs/`: Simülasyon görev ortamları (nesneler, ödül/başarı kriterleri, kameralar).
    *   `teleop/`: Klavye, Gamepad veya SpaceMouse ile insan gösterimi (demonstration) toplama (Teleoperasyonlar/ modülüne bağlı).
    *   `datasets/`: Toplanan simülasyon yörüngelerini `LeRobotDataset` formatına çevirici.
    *   `train/`: LeRobot eğitim başlatıcı ve yapılandırma yönetimi.
    *   `eval/`: Eğitilen ağırlıkların simülasyonda test edilmesi ve başarı metriklerinin ölçülmesi.
    *   `speech/`: Yerel (offline) Türkçe Speech-to-Text (`faster-whisper`), ses yakalama (`sounddevice`) ve kural tabanlı robot komut ayrıştırıcı (`CommandParser`).
*   `scripts/`: URDF doğrulama, ortam testi ve sesli komut canlı test betikleri.

---

## 🎙️ Yerel (Offline) Türkçe Sesli Komut Modülü

MahutKol, tamamen **ücretsiz, yerel ve hafif** çalışan Türkçe Speech-to-Text ve robot komut ayrıştırma altyapısına sahiptir:

1. **STT Motoru:** `faster-whisper` (CUDA GPU'da `float16`, CPU'da `int8` hesaplama).
2. **Ses Yakalama:** `sounddevice` + `numpy` (16 kHz, Mono, Push-to-Talk ve VAD).
3. **Komut Çözücü:** Türkçe dilbilgisi, çekim ekleri ve ManiSkill `MultiShapeRelationalTask` uzamsal ilişkilerine tam uyumlu deterministik parser.

### Hızlı Kullanım ve Test:

```bash
# 1. Doğrudan metin ile komut ayrıştırma testi:
python3 scripts/test_voice_command.py --mode text --input-text "Kırmızı küpü al yeşil silindirin sağına koy"

# 2. Mikrofondan Bas-Konuş (Push-to-Talk) canlı test (Enter ile başlat / durdur):
python3 scripts/test_voice_command.py --mode push_to_talk --model base

# 3. Mikrofondan Sessizlik Tespiti (VAD) ile test:
python3 scripts/test_voice_command.py --mode vad --model base

# 4. Sürekli döngüde çalıştırma:
python3 scripts/test_voice_command.py --mode push_to_talk --interactive
```

---

## 🛠️ Kurulum Adımları (Geliştirici Notu)

1. **Python Ortamı Oluşturma (Conda / Venv):**
   ```bash
   conda create -n mahutkol python=3.10 -y
   conda activate mahutkol
   ```

2. **Bağımlılıkların Yüklenmesi:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Adım Adım Geliştirme Sırası:**
   - [x] CAD modelinden URDF ve mesh doğrulama tamamlandı.
   - [x] SO-100 ve MahutKol kinematik PD kazançları yapılandırıldı.
   - [x] Çok şekilli (`MultiShapeRelationalTask`) 9 nesneli ortam entegre edildi.
   - [x] Yerel Türkçe Speech-to-Text ve Komut Çözücü modülü (`src/mahut_kol/speech/`) tamamlandı.
   - [ ] Teleoperasyon ve sesli komutlarla gösterim (demonstration) toplayın.
   - [ ] LeRobot formatına dönüştürüp ACT/SmolVLA modelini eğitin.
   - [ ] Eğitilmiş modelin kapalı çevrim başarı oranını simülasyonda ölçün.
