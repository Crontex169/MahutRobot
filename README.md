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
*   `scripts/`: URDF doğrulama ve ortam testi için hızlı çalıştırma betikleri.

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
   - [ ] `assets/README.md` kılavuzunu takip ederek CAD modelinizi URDF'e dönüştürün ve `assets/` altına yerleştirin.
   - [ ] `scripts/check_urdf.py` ile URDF ve eklem limitlerinizi doğrulayın.
   - [ ] `src/mahut_kol/robots/mahut_robot.py` dosyasında robot sınıfınızı tamamlayın.
   - [ ] `src/mahut_kol/envs/pick_place_task.py` ile görevi kurgulayın ve `scripts/test_env.py` ile test edin.
   - [ ] `Teleoperasyonlar/record_demos.py` ile insan gösterimi (demonstrations) toplayın.
   - [ ] `src/mahut_kol/datasets/convert_to_lerobot.py` ile veriyi LeRobot formatına dönüştürün.
   - [ ] `src/mahut_kol/train/train_policy.py` ile ACT veya Diffusion Policy modelini eğitin.
   - [ ] `src/mahut_kol/eval/evaluate_policy.py` ile eğitilmiş modelin başarı oranını simülasyonda ölçün.
