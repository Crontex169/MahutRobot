# 🎮 Teleoperasyonlar (MahutKol Teleoperation Suite)

Bu klasör, **MahutKol** 5-DOF robot kolu ve paralel kıskaç sistemi için geliştirilmiş tüm teleoperasyon kodlarını, kontrolcü sınıflarını ve test araçlarını içerir.

---

## 📁 Dosyalar ve Açıklamaları

| Dosya | Açıklama |
|---|---|
| `teleop_maniskill_ps4.py` | **Ana simülasyon teleoperasyon betiği.** ManiSkill 3 ortamında PS4 kolu, genel gamepad veya klavye ile canlı kontrol sağlar (`pd_joint_target_delta_pos` tork korumalı ve sıfır kaymalı). |
| `test_ps4.py` | **Donanım ve gamepad test aracı.** Bağlı kontrolcünün eksen, tetik, D-pad ve buton değerlerini terminale anlık yazdırır. |
| `record_demos.py` | **İnsan gösterimi (demonstration) kaydedici.** Teleoperasyonla robotu kontrol ederken kamera ve eklem yörüngelerini HDF5 formatında kaydeder. |
| `teleop_agent.py` | Kartezyen (End-Effector) uzayında hareket sağlayan Pygame tabanlı `TeleopController` sınıfı. |
| `teleop_ps4.py` | 7-DoF eklem uzayı (`joint0`–`joint4`, kıskaç parmakları) delta pozisyon kontrolü sağlayan `MahutPS4Controller` sınıfı. |

---

## 🚀 Hızlı Kullanım Komutları

### 1. Gamepad / PS4 Kolunu Test Etme
Bağlantıyı doğrulamak ve tuşları görmek için:
```bash
python Teleoperasyonlar/test_ps4.py
```

### 2. Simülasyonda Robotu Teleoperasyon ile Sürme (Varsayılan: MahutKol-PickCube-v1)
```bash
python Teleoperasyonlar/teleop_maniskill_ps4.py
```

Diğer görev ortamlarında çalıştırmak için:
```bash
# PushCube ortamı:
python Teleoperasyonlar/teleop_maniskill_ps4.py --env-id MahutKol-PushCube-v0

# Minimal test ortamı:
python Teleoperasyonlar/teleop_maniskill_ps4.py --env-id MahutKol-Minimal-v0
```

### 3. Teleoperasyon ile İnsan Gösterimi (Demonstration) Toplama
```bash
python Teleoperasyonlar/record_demos.py --env-id MahutKol-PickCube-v1 --device ps4 --episodes 50
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
