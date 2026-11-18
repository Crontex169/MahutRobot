# 🎮 Teleoperasyonlar (MahutKol Teleoperation Suite)

Bu klasör, **MahutKol** 5-DOF robot kolu ve paralel kıskaç sistemi için geliştirilmiş tüm teleoperasyon kodlarını, kontrolcü sınıflarını ve test araçlarını içerir.

---

## 📁 Dosyalar ve Açıklamaları

| Dosya | Açıklama |
|---|---|
| `teleop_maniskill_ps4.py` | **Canlı simülasyon teleoperasyonu (kayıt yok).** Görev ortamında (`MahutKol-MultiShapeRelational-v1`) PS4/gamepad/klavye ile 6 boyutlu mutlak `pd_joint_pos` (`GamepadKeyboardActionSource`). |
| `test_ps4.py` | **Donanım ve gamepad test aracı.** Bağlı kontrolcünün eksen, tetik, D-pad ve buton değerlerini terminale anlık yazdırır. |
| `record_demos.py` | **Basit HDF5 kaydedici.** Başarılı bölümlerin eylem (6) ve state (6) yörüngelerini HDF5'e ekler. VLA verisi için asıl yol `scripts/record_vla_demos.py`. |
| `teleop_agent.py` | Kartezyen (End-Effector) uzayında hareket sağlayan Pygame tabanlı `TeleopController` sınıfı. |
| `teleop_ps4.py` | Gamepad/klavye okuyucu `MahutPS4Controller` (7'li delta; `GamepadKeyboardActionSource` 6 boyutlu eyleme çevirir). |

---

## 🚀 Hızlı Kullanım Komutları

### 1. Gamepad / PS4 Kolunu Test Etme
Bağlantıyı doğrulamak ve tuşları görmek için:
```bash
python Teleoperasyonlar/test_ps4.py
```

### 2. Simülasyonda Robotu Teleoperasyon ile Sürme (görev ortamı)
```bash
python Teleoperasyonlar/teleop_maniskill_ps4.py [--seed 0]
```

### 3. Teleoperasyon ile İnsan Gösterimi (Demonstration) Toplama
```bash
python Teleoperasyonlar/record_demos.py --output-path data/demos/raw_demos.h5 --num-episodes 50
# VLA (LeRobot) verisi için: python scripts/record_vla_demos.py --device gamepad
```

---

## 🕹️ Kontrol Tuş Haritası (PS4 & Klavye)

### PS4 Kolu (DualShock 4 / Genel Gamepad)
- **Sol Stick (X / Y):** Taban Yaw (`joint0`) ve Omuz Pitch (`joint1`)
- **Sağ Stick (Y / X):** Dirsek Pitch (`joint2`) ve Bilek Pitch (`joint3`)
- **D-Pad (Sol / Sağ):** Bilek Roll (`joint4`)
- **R2 Tetiği:** Kıskaç Kapat (Holding force korunur)
- **L2 Tetiği:** Kıskaç Aç
- **Üçgen (Triangle) / Options:** Ortamı / Epizotu Sıfırla (Reset)

### Klavye (Yedek Kontrol)
- **A / D:** Taban Dönüşü (Yaw)
- **W / S:** Omuz Eğimi (Pitch)
- **Yukarı / Aşağı Ok Tuşları:** Dirsek Eğimi
- **Sol / Sağ Ok Tuşları:** Bilek Eğimi
- **Q / E:** Bilek Yuvarlanma (Roll)
- **Boşluk (Space):** Kıskacı Kapat
- **F Tuşu:** Kıskacı Aç
- **R Tuşu:** Sıfırla (Reset)
