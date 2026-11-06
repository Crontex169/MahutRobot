# 🔌 MahutKol Fiziksel Donanım Spesifikasyonları ve Sim-to-Real Kılavuzu

Bu belge, **MahutKol** robotunun fiziksel prototipinde kullanılan elektronik ve elektromekanik bileşenlerin teknik özelliklerini, matematiksel hesaplamalarını ve bu donanımların simülasyon ortamındaki (ManiSkill 3 / MuJoCo) karşılıklarını detaylandırır.

---

## 1. 🦾 Aktüatörler: Feetech FT5330M 35Kg Digital Servo

Robotun 5 kol eklemi ve paralel kıskaç mekanizmasında yüksek torklu **Feetech FT5330M** dijital servo motorlar kullanılmaktadır.

### Teknik Veriler:
* **Çalışma Gerilimi:** 6.0V - 7.4V (Tavsiye edilen: 7.4V 2S LiPo veya regüle güç kaynağı)
* **Kilitlenme Torku (Stall Torque @ 7.4V):** $35.0\text{ kg}\cdot\text{cm}$
* **Uluslararası Tork Değeri (SI):**
  $$\tau = 35.0\text{ kg}\cdot\text{cm} \times 0.0980665 = \mathbf{3.432\text{ N}\cdot\text{m}}$$
* **Boştaki Dönüş Hızı (@ 7.4V):** $0.18\text{ sn} / 60^\circ$ ($\approx 5.8\text{ rad/s}$)
* **Kilitlenme Akımı (Stall Current):** $\approx 3.5\text{A} - 4.0\text{A}$
* **Boşta Çalışma Akımı:** $\approx 150\text{ mA}$
* **Dişli Yapısı:** Güçlendirilmiş Çelik / Bakır Metal Dişli
* **Geri Bildirim / Kontrol:** Standart 50Hz PWM veya Seri Bus (model varyantına göre)

### Simülasyondaki Karşılığı:
Kol eklemleri (`joint0` – `joint4`) için `configs/robot/mahut_kol.yaml` ve `src/mahut_kol/robots/mahut_robot.py` dosyalarında `force_limit` değeri doğrudan motorun stall torku olan **`3.432 Nm`** olarak yapılandırılmıştır.

---

## 2. 🧠 Ana Kontrol Kartı (MCU): ESP32-S3-N16R8

Robotun tüm eklem yörünge kontrolü, ters kinematik çözümleri ve teleoperasyon haberleşmesi ESP32-S3 çipi üzerinde koşacaktır.

### Teknik Veriler:
* **İşlemci:** Xtensa® 32-bit LX7 Çift Çekirdek (240 MHz frekans)
* **Flash Bellek:** 16 MB SPI Flash (`N16`)
* **PSRAM:** 8 MB Octal PSRAM (`R8`) — *Büyük yörünge tamponları ve kinematik matrisler için*
* **Kablosuz Bağlantı:** 2.4 GHz Wi-Fi (802.11 b/g/n) & Bluetooth 5 (LE / Mesh)
* **Arayüzler:** USB OTG (Yerleşik programlama ve doğrudan PC teleoperasyon akışı), MCPWM (Yüksek hassasiyetli servo motor donanım zamanlayıcıları), UART, I2C, SPI.
* **Besleme:** 3.3V mantık seviyesi (LDO veya buck converter ile harici 5V/7.4V'tan indirilir).

---

## 3. 🦀 Kıskaç Mekaniği (Gripper-v2): Kramayer ve Pinyon Hesabı

Kıskaç, servonun dairesel dönüşünü lineer çift yönlü harekete çeviren pinyon dişli ve çift kramayer mekanizmasına sahiptir.

### Dişli Parametreleri:
* **Dişli Modülü:** $m = 0.8\text{ mm}$
* **Diş Sayısı:** $z = 18$
* **Bölüm Dairesi Çapı:** $D = m \times z = 0.8 \times 18 = 14.4\text{ mm}$
* **Dişli Yarıçapı:** $r = 7.2\text{ mm} = 0.0072\text{ m}$

### Teorik vs. Gerçek Kavrama Kuvveti:
1. **Teorik Stall Kuvveti:**
   Motor sonuna kadar kilitlendiğinde kramayer dişliye aktarılacak teorik itme kuvveti:
   $$F_{\text{teorik}} = \frac{\tau}{r} = \frac{3.432\text{ N}\cdot\text{m}}{0.0072\text{ m}} \approx \mathbf{476.7\text{ N}} \quad (\approx 48.6\text{ kgf})$$

2. **Gerçek Dünyadaki Çalışma Koşulu (Neden 477N Uygulanmaz?):**
   * **3D Baskı Dayanımı:** 48 kg'lık baskı, 3D baskı PLA/PETG dişlileri sıyırır ve parmakları kırar.
   * **Aşırı Akım / Stall Koruması:** Motor stall akımına girdiğinde 3.5A çeker. Bu durum bataryayı hızla tüketir ve motor sargılarını yakar.
   * **Hassas Tutuş:** Masadaki 15-50 gramlık bir küpü kavramak için gereken gerçek kuvvet **$2\text{ N} - 8\text{ N}$** arasındadır.
   * **Yazılımsal Tork Sınırı:** ESP32 yazılımında kıskaç için akım sınırlaması veya pozisyon eşiği uygulanır.

3. **Sim-to-Real Dengelemesi:**
   Simülatörde parçaların sonsuz rijit (sıfır esneklik) olmasından kaynaklanan "fırlama ve sarsıntı" etkilerini engellemek ve donanım esnekliğini temsil etmek için:
   * Kıskaç `force_limit`: **`15.0 N`** ($\approx 1.5\text{ kgf}$ emniyetli sıkıştırma)
   * Kıskaç `stiffness`: **`150.0`** (Plastik esnekliğini yansıtan yumuşak yay katsayısı)
   * Kıskaç `damping`: **`15.0`**

---

## 4. 📐 Fiziksel Montaj ve Koordinat Referansı

* **Masaüstü Montajı:** Robotun taban flanşı masanın üzerine oturtulduğunda, motor gövdesinin alt kısmı tabla flanş yüksekliği kadar zeminden yukarıda kalır.
* **Simülasyon Z Kotu:** Robot simülasyonda masanın sıfır seviyesine gömülmemeli, MuJoCo modelinde olduğu gibi montaj flanşı kalınlığı kadar ($z \approx 0.0558\text{ m}$) yukarıda başlatılmalıdır.
