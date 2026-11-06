# 📚 MahutKol Dokümantasyon Merkezi

Bu klasör, projenin simülasyon ve öğrenme altyapısını oluşturan **ManiSkill 3** ve **Hugging Face LeRobot** kütüphanelerinin resmi dokümantasyon referanslarını, mimari şablonlarını ve entegrasyon kılavuzlarını içerir.

---

## 📑 Belgeler

1. **[ManiSkill 3 Geliştirici ve Mimari Kılavuzu](MANISKILL_GUIDE.md)**
   * PhysX & SAPIEN GPU simülasyon yapısı
   * `BaseAgent` ile özel robot tanımlama ve PD kontrolcüler
   * Görev tasarımı (`BaseEnv`), kameralar ve gözlem modları
   * Resmi dokümantasyon ve API bağlantıları

2. **[Hugging Face LeRobot Geliştirici Kılavuzu](LEROBOT_GUIDE.md)**
   * `LeRobotDataset` veri mimarisi (Parquet + MP4 / Zarr)
   * Desteklenen modeller: ACT (Action Chunking) ve Diffusion Policy
   * Eğitim komutları (CLI ve Python API)
   * Kapalı çevrim değerlendirme (`eval.py`) ve donanım gereksinimleri

3. **[VR (Sanal Gerçeklik) Teleoperasyon Kılavuzu](VR_TELEOPERATION_GUIDE.md)**
   * Meta Quest (Quest 2/3/Pro) entegrasyonu
   * ManiSkill 3 ALVR/WebXR 6-DoF el takibi ve stereoskopik görüntü akışı
   * LeRobot için doğrudan VR ile gösterim (demonstration) toplama

4. **[PS4 Kolu / Gamepad ile Teleoperasyon Kılavuzu](GAMEPAD_TELEOP_GUIDE.md)**
   * DualShock 4 ve standart gamepad eşleştirmesi (Mapping)
   * 6-DoF End-Effector ve analog tetik (R2/L2) kontrolü
   * Linux USB & Bluetooth bağlantı adımları

5. **[ManiSkill & LeRobot Entegrasyon Hattı](MAHUTKOL_PIPELINE.md)**
   * CAD ➔ URDF ➔ ManiSkill ➔ LeRobot ➔ Değerlendirme veri hattı
   * Projedeki Python dosyalarının kütüphaneler ile eşleşmesi

6. **[Fiziksel Donanım ve Sim-to-Real Spesifikasyonları](DONANIM_BILGILERI.md)**
   * Feetech FT5330M (35 kg·cm) motor ve ESP32-S3-N16R8 kontrolcü
   * Kıskaç pinyon/kramayer mekaniği ve tork/kuvvet hesapları
   * Simülasyon kararlılık parametreleri ve Sim-to-Real köprüsü
