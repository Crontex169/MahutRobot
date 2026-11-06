# SO-100 Robot Kolu İçin Web Kamerası (MediaPipe Hands) Teleoperasyon Modülü

Bu sistem, bilgisayarınızın web kamerasını (**OpenCV** + **MediaPipe Hands**) kullanarak el hareketlerinizle MuJoCo simülasyonundaki **SO-100** robot kolunu ultra düşük gecikme ve yüksek ergonomiyle gerçek zamanlı kontrol etmenizi sağlar.

Mevcut WebXR (telefon) teleoperasyon altyapısına dokunulmadan, tamamen bağımsız ve modüler olarak eklenmiştir.

---

## 1. Gecikme ve Kontrol İyileştirmeleri (Neler Değişti?)

1. **Sıfır Tampon Gecikmesi (Zero-Latency Threaded Video Capture):**
   - Eski sistemde OpenCV `cap.read()` tek iş parçacığında çalıştığı için Linux V4L2 sürücü tamponunda kareler birikiyor ve 500ms - 1000ms gecikme ("ping") oluşuyordu.
   - Yeni `ThreadedCamera` arka plan iş parçacığıyla kareleri sürekli okuyarak eski kareleri anında eler; döngüye daima fiziksel anın en güncel karesi mikrosaniye içinde iletilir.
2. **1:1 Gerçek Zamanlı Fizik Alt-Adımları (Multi-Substepping):**
   - Eski sistemde kamera karesi başına sadece 1 fizik adımı ($dt=0.002\text{s}$) çalışıyordu. Bu nedenle simülasyon fiziksel zamandan 16 kat daha yavaş ilerliyor ve robot motorları hantal, gecikmeli tepki veriyordu.
   - Yeni sistemde iki kare arasında geçen süreye göre (~33ms) dinamik olarak ~16 fizik adımı işletilir. Robot kolu elinizi anlık ve birebir takip eder.
3. **Ergonomik Çift Modlu Debriyaj (Toggle / Hold):**
   - **Toggle Modu (Varsayılan - Önerilen):** Klavyede `Space` tuşuna veya pencereye fareyle **bir kez basıldığında** debriyaj açılır (**YEŞİL**). Tuşu basılı tutmak zorunda kalmadan elinizi rahatça hareket ettirirsiniz. Tekrar `Space` bastığınızda robot konumunu kilitler (**KIRMIZI**).
   - **Hold Modu:** İsteğe bağlı olarak `Space` tuşuna basılı tutularak kullanılır; Linux'un 500ms tuş tekrar gecikmesini tolere edecek şekilde güncellenmiştir.
   - **[M] Tuşu:** İki mod arasında anında geçiş yapar.
4. **Kararlı 3B Derinlik (İleri/Geri) ve Ölü Bölge (Deadband):**
   - Elinizi sağa/sola çevirdiğinizde 2B mesafelerin kısalmasından ötürü robotun geriye kaçması engellenmiştir. Hipotenüs tabanlı avuç ölçeği ve %3 ölü bölge ile derinlik kontrolü son derece pürüzsüz hale getirilmiştir.
5. **Bilek Eğimi/Dönüşü Aç/Kapa ([W] Tuşu):**
   - 5-DoF kinematik kısıtlarından ötürü istenirse `W` tuşu ile bilek açı takibi devre dışı bırakılıp yalnızca 3B konum kontrolü yapılabilir.

---

## 2. Kurulum ve Gereksinimler

```bash
pip install mediapipe opencv-python mujoco numpy
```

---

## 3. Çalıştırma Komutları

`Side_project` klasörüne gidin ve scripti çalıştırın:

```bash
cd Side_project
python3 teleop_webcam.py
```

### Ek Parametreler (Opsiyonel)

- **Farklı kamera seçimi:**
  ```bash
  python3 teleop_webcam.py --camera 1
  ```
- **Hold (basılı tutma) debriyaj modu ile başlatma:**
  ```bash
  python3 teleop_webcam.py --hold-clutch
  ```
- **Kamera aynalama (mirror) modunu kapatma:**
  ```bash
  python3 teleop_webcam.py --no-mirror
  ```

---

## 4. Kontrol Mantığı ve Kullanım

### A. Debriyaj (Clutch)
- **Space veya Fare Sol Tık:** Debriyajı açar/kapatır (Toggle).
- **[M] Tuşu:** Toggle $\leftrightarrow$ Hold modunu değiştirir.

### B. Robot Eksen Haritalaması
- **Sağ / Sol ($\Delta X_{\text{robot}}$):** Elinizi kamerada sağa veya sola hareket ettirin.
- **Yukarı / Aşağı ($\Delta Z_{\text{robot}}$):** Elinizi kamerada yukarı kaldırın veya aşağı indirin.
- **İleri / Geri - Derinlik ($\Delta Y_{\text{robot}}$):**
  - Elinizi kameraya **yaklaştırdığınızda** robot ileri uzanır ($-Y$).
  - Elinizi kameradan **uzaklaştırdığınızda** robot geriye çekilir ($+Y$).

### C. Kıskaç (Gripper) Kontrolü
- Başparmak ucu (4) ile İşaret parmağı ucu (8) arası mesafe avuç içi boyutuyla normalize edilir.
- **Pinch (Kapatma):** İki parmağınızı yaklaştırdığınızda kıskaç yumuşakça kapanır (`-1.0`).
- **Açma:** Parmaklarınızı ayırdığınızda kıskaç açılır (`1.0`).

### D. Bilek Kontrolü ([W] Tuşu)
- `W` tuşuna basarak bilek eğimi ve dönüşünü (Pitch/Roll) aktifleştirebilir veya sabit tutabilirsiniz.

---

## 5. Kısayol Tuşları Tablosu

| Tuş | İşlev |
| :--- | :--- |
| **`Space`** veya **`Sol Tık`** | Debriyajı Aç / Kapat (Toggle) |
| **`M`** | Debriyaj Modunu Değiştir (Toggle / Hold) |
| **`W`** | Bilek Açısı Takibini Aç / Kapat (Pitch & Roll) |
| **`R`** | Robotu Başlangıç Pozisyonuna Sıfırla |
| **`Q`** veya **`ESC`** | Simülasyonu ve Kamerayı Kapat |

---

## 6. Birim Testleri (Unit Tests)

```bash
cd Side_project
python3 -m unittest test_webcam_teleop.py
```
