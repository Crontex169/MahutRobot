# SO-100 Robot Kolu WebXR (ARCore) Telefon Teleoperasyon Sistemi

Bu sistem, Android akıllı telefonunuzu (örn. **Poco X7 Pro**) uzamsal bir 6-DoF sanal kumandaya dönüştürerek **MuJoCo** simülasyonundaki **SO-100** (5-DoF + 1 Gripper) robot kolunu gerçek zamanlı kontrol etmenizi sağlar.

Telefona harici hiçbir APK veya uygulama yüklemeden, doğrudan mobil Google Chrome tarayıcısındaki **WebXR Device API (`immersive-ar` / ARCore)** üzerinden çalışır.

---

## 1. Mimarisi ve Çalışma Mantığı

1. **WebXR ARCore VIO (Görsel Eylemsizlik Navigasyonu):**
   Telefonun kamera ve IMU sensörleri ile uzaydaki 6 serbestlik dereceli konumu $(X, Y, Z)$ ve kuaterniyon yönelimi $(q_x, q_y, q_z, q_w)$ 60 Hz frekansta okunur.
2. **USB Port Köprüleme (`adb reverse`):**
   Telefon USB kablosuyla bilgisayara bağlandığında `adb reverse tcp:8000 tcp:8000` yapılarak telefonun `localhost:8000` portu bilgisayara köprülenir.
   * `http://localhost:*` tarayıcılar tarafından **Secure Context** (Güvenli Bağlam) kabul edildiği için SSL/HTTPS sertifikası gerekmez.
   * USB üzerinden yerel döngü (loopback) ile ağ gecikmesi **<5ms** seviyesindedir.
3. **Asenkron Arka Plan Sunucusu (FastAPI + Uvicorn):**
   MuJoCo'nun ana fizik ve görüntüleme döngüsünü engellememek için WebSocket sunucusu bağımsız bir arka plan daemon thread'inde çalışır.
4. **5-DoF Nullspace Projeksiyonlu Damped Least Squares (DLS) IK:**
   SO-100 5 eklemli bir kol olduğundan tam 6-DoF kısıtlama matematiksel kilitlenmeye yol açar. Geliştirilen IK çözücü, 3B uç nokta konumunu $(X, Y, Z)$ birincil öncelikle tam çözer; bilek eğimini ($Pitch$) ve dönüşünü ($Roll$) ise nullspace içinde pozisyonu bozmadan takip eder.
5. **Debriyaj (Clutch) Mekanizması:**
   Ekranın alt yarısındaki büyük debriyaj alanına parmak basılı tutulduğu sürece o anki telefon pozisyonu referans (anchor) alınarak bağıl hareket robota aktarılır. Parmak kaldırıldığında robot olduğu yerde kilitlenir; böylece kullanıcı elini havada rahat bir konuma geri getirip teleoperasyona devam edebilir (Fareyi kaldırıp masada yeniden konumlandırmak gibi).
6. **Titreme Önleme (Smoothing) ve Güvenlik:**
   El titremelerini sönümlemek için Üstel Hareketli Ortalama (EMA) filtresi, AR takibinin anlık kopmalarına karşı sıçrama kırpıcı (step limiter) ve robotun çalışma hacmini sınırlandıran güvenlik kafesi (workspace bounding box) mevcuttur.

---

## 2. Gereksinimler ve Kurulum

Gerekli Python paketleri:
```bash
pip install mujoco fastapi uvicorn websockets numpy scipy
```

Telefon tarafında:
- **Android Telefon** (ARCore / Google Play Hizmetleri for AR destekli, örn. Poco X7 Pro).
- **Google Chrome** tarayıcısı.
- **USB Hata Ayıklama (USB Debugging)** açık olmalıdır:
  1. *Ayarlar -> Telefon Hakkında -> Yapım Numarası (Build Number)* üzerine 7 kez dokunun.
  2. *Ayarlar -> Ek Ayarlar -> Geliştirici Seçenekleri -> USB Hata Ayıklama* seçeneğini açın.

Bilgisayar tarafında `adb` (Android Debug Bridge) kurulu olmalıdır:
```bash
# Ubuntu/Debian için (eğer kurulu değilse):
sudo apt install adb
```

---

## 3. Sistemi Çalıştırma Adımları

### 1. Adım: Telefonu USB ile PC'ye Bağlayın ve Portu Köprüleyin
Telefonu USB kablosuyla bağlayın (telefonda çıkan "Bu bilgisayara güvenilsin mi?" uyarısını onaylayın). Ardından terminalde şu komutu çalıştırın:

```bash
adb reverse tcp:8000 tcp:8000
```
*(Bu komut telefonun `localhost:8000` isteklerini doğrudan PC'deki 8000 portuna yönlendirir).*

### 2. Adım: Teleoperasyon Betiğini Başlatın
Terminalde `Side_project` klasörüne gidin ve betiği çalıştırın:

```bash
cd /home/deniz/Ortak/MahutKol/Side_project
python3 webxr_teleop.py
```
*Ekranda MuJoCo penceresi açılacak ve SO-100 robot kolu ile hedefi temsil eden yeşil bir işaretçi (`mocap target`) belirecektir.*

### 3. Adım: Telefonda Tarayıcıyı Açın
Telefondaki Google Chrome tarayıcısında adres satırına şunu yazıp girin:
```text
http://localhost:8000
```

### 4. Adım: Teleoperasyona Başlayın
1. Ekranda çıkan mavi **"🥽 AR Teleop Başlat (ARCore)"** butonuna dokunun.
2. Kamera iznini onaylayın.
3. Alt kısımdaki büyük **"DEBRİYAJ (BASILI TUTUN)"** butonuna basılı tutarak telefonunuzu havada hareket ettirin:
   - Telefonu ileri/geri hareket ettirdiğinizde robot kolu ileri/geri uzanır.
   - Telefonu sağa/sola hareket ettirdiğinizde robot kolu sağa/sola döner.
   - Telefonu yukarı/aşağı kaldırdığınızda robot kolu yükselir/alçalır.
   - Telefonu eğip büktüğünüzde bilek pitch ve roll eksenleri takip eder.
4. Kolunuz rahatsız bir açıya geldiğinde parmağınızı debriyajdan çekin, elinizi rahat bir konuma getirin ve tekrar basılı tutarak devam edin.
5. Kıskacı açıp kapatmak için ekrandaki **"Kıskaç (Gripper)"** kaydırıcısını kullanın.
6. Sahneyi başlangıç pozuna döndürmek için sağ üstteki **"↺ Sıfırla (Reset)"** butonuna basın.

---

## 4. Yapılandırma ve Eksen Ayarları (`CONFIG`)

[`webxr_teleop.py`](file:///home/deniz/Ortak/MahutKol/Side_project/webxr_teleop.py) dosyasının en üstünde yer alan `CONFIG` sözlüğü üzerinden hassasiyet ve eksen yönlerini ihtiyacınıza göre kolayca değiştirebilirsiniz:

```python
CONFIG = {
    # Hassasiyet ve Ölçek Çarpanları
    "pos_scale": 1.0,           # Telefon yer değiştirmesi (m) -> Robot hedefi (m)
    "rot_scale": 1.0,           # Telefon eğimi (rad) -> Robot bilek açısı (rad)

    # Eksen Eşleşmesi (WebXR Telefon Ekseni -> MuJoCo Robot Ekseni)
    # 0: X (Sağ), 1: Y (Yukarı), 2: Z (Geri / -Z İleri)
    "axis_map": {
        "x": 2,  # Robot X ekseni <- Telefon Z ekseni
        "y": 0,  # Robot Y ekseni <- Telefon X ekseni
        "z": 1,  # Robot Z ekseni <- Telefon Y ekseni
    },
    "axis_signs": {
        "x": -1.0,  # -Z_telefon -> +X_robot
        "y": -1.0,  # -X_telefon -> +Y_robot
        "z": 1.0,   # +Y_telefon -> +Z_robot
    },

    # Titreme Önleme (Smoothing) ve Güvenlik Sınırları
    "ema_alpha_pos": 0.30,      # Pozisyon EMA filtresi (0.1: çok yumuşak, 1.0: anlık)
    "ema_alpha_rot": 0.30,      # Açı EMA filtresi
    "max_step_pos": 0.035,      # Tek bir karede izin verilen azami konum sıçraması (m)
    "workspace_limits": {
        "x": (-0.28, 0.28),     # Robot sağ/sol sınırları (metre)
        "y": (-0.48, -0.05),    # Robot ileri/geri sınırları (metre)
        "z": (0.015, 0.42),     # Zemin üstü çalışma yüksekliği (metre)
    },

    # Fizik ve Kontrol Döngüsü
    "ik_iterations": 4,         # Her adımda DLS IK çözücü iterasyonu
    "loop_rate_hz": 120.0,      # Simülasyon döngü frekansı (Hz)
}
```

---

## 5. Dosya Yapısı

```
Side_project/
├── index.html                   # WebXR ARCore mobil arayüzü (DOM-overlay, debriyaj, gripper slider)
├── ik_solver.py                 # SO-100 için 5-DoF Nullspace-Projected DLS Ters Kinematik çözücü
├── webxr_teleop.py              # FastAPI WebSocket sunucusu ve MuJoCo teleoperasyon ana scripti
├── view_so100.py / main.py      # Bağımsız pasif model görüntüleyici
├── README.md                    # Genel proje açıklaması
├── README_TELEOP.md             # Bu ayrıntılı teleoperasyon rehberi
└── so_arm100/
    ├── teleop_scene.xml         # Mocap hedef küresi içeren teleoperasyon sahnesi
    ├── scene.xml                # Standart sahne dosyası
    ├── so_arm100.xml            # SO-100 MJCF robot modeli (ee_site tanımlı)
    └── assets/                  # 3D STL mesh dosyaları
```
