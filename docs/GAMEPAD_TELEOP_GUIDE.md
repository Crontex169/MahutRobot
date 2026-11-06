# 🎮 PS4 Kolu (DualShock 4) ve Gamepad ile Teleoperasyon Kılavuzu

Evet, **PlayStation 4 (DualShock 4)** veya standart herhangi bir USB/Bluetooth gamepad ile robot kolunu simülasyonda doğrudan teleopere edebilir ve insan gösterimi (demonstration) verisi toplayabilirsiniz.

DualShock 4, Linux ortamında çekirdek düzeyinde (`hid-playstation` / `hid-sony`) tak-çalıştır olarak doğrudan tanınır ve `pygame.joystick` üzerinden sıfır ek yapılandırma ile çalışır.

---

## 🕹️ PS4 Kolu Tuş ve Eksen Haritası (Mapping)

Robotun uç organı (End-Effector) uzayda 6-DoF serbestlik derecesiyle sürülür:

| PS4 Butonu / Ekseni | Robot Eylemi (Action) | Açıklama |
| :--- | :--- | :--- |
| **Sol Analog (İleri / Geri)** | **X Ekseni (`dx`)** | Robot uç organını ileri ve geri hareket ettirir. |
| **Sol Analog (Sol / Sağ)** | **Y Ekseni (`dy`)** | Robot uç organını sola ve sağa kaydırır. |
| **Sağ Analog (Yukarı / Aşağı)** | **Z Ekseni (`dz`)** | Robot uç organını yukarı kaldırır / aşağı indirir. |
| **Sağ Analog (Sol / Sağ)** | **Yaw Dönüşü (`dyaw`)** | Uç organı kendi ekseni etrafında sağa/sola döndürür. |
| **D-Pad (Yukarı / Aşağı)** | **Pitch Açısı (`dpitch`)** | Tutucunun eğilme açısını (bilek eğimi) ayarlar. |
| **D-Pad (Sol / Sağ)** | **Roll Açısı (`droll`)** | Tutucunun eksenel yuvarlanma açısını ayarlar. |
| **R2 Tetiği (Trigger)** | **Gripper Kapat (Kavrama)** | Orantılı basış: Küpü yakalamak için tutucuyu sıkar. |
| **L2 Tetiği (Trigger)** | **Gripper Aç (Bırakma)** | Orantılı basış: Tutucuyu açarak küpü bırakır. |
| **Options / Üçgen (△)** | **Bölümü Sıfırla (Reset)** | Başarısız hamlede epizotu anında yeniden başlatır. |
| **Çarpı (X / Cross)** | **Bölümü Onayla ve Kaydet** | Görev başarılı bittiğinde veriyi diske kaydeder. |

---

## 🔌 Linux'ta PS4 Kolunu Bağlama

### Yöntem 1: USB Kablosu ile (Önerilen - Sıfır Gecikme)
1. PS4 kolunu micro-USB kablosu ile bilgisayara takın.
2. Kolun ışığı yanacak ve Linux anında `/dev/input/js0` olarak tanıtacaktır.
3. Kontrol etmek için terminalde şu komutu çalıştırabilirsiniz:
   ```bash
   ls -l /dev/input/js*
   ```

### Yöntem 2: Bluetooth (Kablosuz)
1. PS4 kolunu eşleştirme moduna alın: **Share** ve **PS** tuşlarına ışık çubuğu beyaz renkte hızlıca yanıp sönene kadar basılı tutun.
2. Linux Bluetooth ayarlarından "Wireless Controller" cihazını seçip eşleştirin.

---

## 🚀 Çalıştırma

Gösterim toplama betiğini doğrudan PS4 kolu seçeneğiyle başlatabilirsiniz:

```bash
python src/mahut_kol/teleop/record_demos.py --device ps4 --num-episodes 50
```

> **Not:** Eğer kol bağlı değilse veya pil biterse, sistem otomatik olarak klavye kontrolüne (`W/S/A/D/Q/E`) geri düşer.
