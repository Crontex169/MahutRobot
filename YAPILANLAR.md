# 📋 MahutKol Projesinde Yapılanlar (Changelog & Aktivite Günlüğü)

Bu dosya, projenin başlangıcından bu yana gerçekleştirilen tüm araştırma, dokümantasyon, kodlama, donanım entegrasyonu ve hata giderme adımlarını kronolojik ve kategorik olarak özetler.

---

## 1. 📚 Proje İncelemesi ve Dokümantasyon Kurulumu
* **Kod Tabanı ve STL Analizi:**
  * `assets/tasarım/` altındaki 22 STL parçası incelendi; parçaların **SO-100 (SO-ARM100)** 6-DOF açık kaynaklı robot koluna ait olduğu tespit edildi.
  * Mevcut iskelet kodlar (`robots/`, `envs/`, `teleop/`, `datasets/`, `train/`, `eval/`) tarandı.
* **Resmi Kütüphane Kılavuzları Oluşturuldu (`docs/`):**
  * `docs/MANISKILL_GUIDE.md`: ManiSkill 3 / SAPIEN GPU simülasyonu, `BaseAgent`, kontrolcüler ve görev yapısı belgelendi.
  * `docs/LEROBOT_GUIDE.md`: Hugging Face LeRobot veri mimarisi (`LeRobotDataset`), ACT ve Diffusion Policy eğitim kılavuzu hazırlandı.
  * `docs/MAHUTKOL_PIPELINE.md`: CAD ➔ URDF ➔ Simülasyon ➔ LeRobot ➔ Değerlendirme veri akışı şemalaştırıldı.
  * `docs/VR_TELEOPERATION_GUIDE.md`: Meta Quest 2/3/Pro başlıkları ile 6-DoF el takibi ve ALVR/WebXR akış rehberi eklendi.
  * `docs/GAMEPAD_TELEOP_GUIDE.md`: PS4 DualShock 4 ve genel joystick tuş haritası belgelendi.
  * `docs/README.md`: Dokümantasyon merkezi için ana içindekiler tablosu oluşturuldu.

---

## 2. 🎮 PS4 Kolu (DualShock 4) ve Teleoperasyon Entegrasyonu
* **Kontrolcü Mantığı Yazıldı ([teleop_agent.py](src/mahut_kol/teleop/teleop_agent.py)):**
  * Pygame tabanlı `TeleopController` sınıfı sıfırdan implemente edildi.
  * Sol Analog (X/Y), Sağ Analog (Z/Yaw), D-Pad (Pitch/Roll) ve L2/R2 tetikleri (Gripper açma/kapatma) haritalandı.
  * Analog çubukların doğal titremelerini (drift) sıfırlamak için `deadzone = 0.08` filtresi entegre edildi.
  * Üçgen/Options tuşlarına "Bölüm Sıfırlama (Reset)", Çarpı (X) tuşuna "Başarılı Kayıt" fonksiyonları atandı.
* **Gösterim Kaydedici Tamamlandı ([record_demos.py](src/mahut_kol/teleop/record_demos.py)):**
  * Daha önce `NotImplementedError` olan `record_demonstrations` döngüsü dolduruldu.
  * Aksiyonları ortama iletme, epizotları tamponlama ve başarılı gösterimleri HDF5 formatında diske yazma altyapısı kuruldu.
* **Donanım Test Aracı Eklendi ([scripts/test_ps4.py](scripts/test_ps4.py)):**
  * Simülasyondan bağımsız, bağlı gamepad cihazlarını tespit edip anlık analog ve buton değerlerini terminalde canlı gösteren test betiği yazıldı ve kullanıcının koluyla doğrulandı.

---

## 3. 🤖 SO-100 (SO-ARM100) Robot Entegrasyonu
* **URDF Dosyası Üretildi ([assets/urdf/so100.urdf](assets/urdf/so100.urdf)):**
  * Robotun 6 eklemi (`Rotation`, `Pitch`, `Elbow`, `Wrist_Pitch`, `Wrist_Roll`, `Jaw`) doğru eksenler, atalet matrisleri ve hareket limitleriyle URDF formatında tanımlandı.
* **Konfigürasyonlar Güncellendi:**
  * [configs/robot/mahut_kol.yaml](configs/robot/mahut_kol.yaml) dosyası `so_arm100` ismi ve 6 eklemli yapıya uyarlandı.
  * [src/mahut_kol/robots/mahut_robot.py](src/mahut_kol/robots/mahut_robot.py) sınıfı SO-100 URDF ve eklem listeleriyle eşleştirildi.

---

## 4. 🕹️ İnteraktif 3D Simülasyon Araçları Geliştirildi
* **Hafif 3D Simülatör ([scripts/sim_so100_ps4.py](scripts/sim_so100_ps4.py)):**
  * Harici ağır fizik motoru gerekmeden, sadece `pygame` ile 60 FPS çalışan 3D projeksiyon motoru yazıldı.
  * SO-100 ileri kinematik modeli (FK), hareketli eklemler, çift parmaklı kıskaç, zemin ızgarası ve hedef küp çizildi.
  * Fare ile 360° kamera döndürme ve yakınlaştırma (orbit/zoom) eklendi.
  * Ekranın köşesinde her eklemin anlık açısını derece cinsinden gösteren HUD paneli kuruldu.
* **ManiSkill Yerleşik Ortam Teleoperasyonu ([scripts/teleop_maniskill_ps4.py](scripts/teleop_maniskill_ps4.py)):**
  * ManiSkill 3'ün resmi olarak desteklediği yerleşik `PickCubeSO100-v1` ve `PickCube-v1` (Franka Panda) ortamlarını doğrudan PS4 koluyla kontrol eden betik yazıldı.

---

## 5. 🛠️ Giderilen Hatalar ve Uyumluluk Düzeltmeleri
1. **Pygame Joystick Metot İsimleri:**
   * Pygame 2.x API standardına uygun olarak `get_num_axes()`, `get_num_buttons()`, `get_num_hats()` çağrıları `get_numaxes()`, `get_numbuttons()`, `get_numhats()` şeklinde düzeltildi.
2. **NumPy 2.x / OpenCV Çakışması:**
   * OpenCV (`cv2`) ve ManiSkill C-API'sinin NumPy 2.x (`2.2.6`) ile yaşadığı `_ARRAY_API not found` hatası analiz edildi.
   * [requirements.txt](requirements.txt) ve [pyproject.toml](pyproject.toml) dosyalarında `numpy<2` kuralı getirilerek çözüm belgelendi.

---

## 6. 🧩 Model ve Varlıkların Tek Dosyada / Bütünleşik Yapıda Birleştirilmesi
* **Tek Bağımsız Sahne Modeli ([assets/mahut_kol_scene.xml](assets/mahut_kol_scene.xml)):**
  * `Kol_model/robot.xml` ve `scene.xml` dosyaları hiçbir dış `<include>` bağımlılığı kalmayacak şekilde tek bir MuJoCo MJCF dosyasında birleştirildi.
  * Kameralar (`cam_top`, `cam_side`), çalışma masası, kırmızı hedef küp, yeşil bırakma kutusu ve 4-DOF + Gripper-v2 robot gövdesi tek çatı altında toplandı.
* **Tek Bağımsız Robot Modeli ([assets/mahut_kol_robot.xml](assets/mahut_kol_robot.xml)):**
  * Ortamdan bağımsız sadece robotu içeren MJCF modeli üretildi.
* **Gerçek Mesh'li URDF Üretildi ([assets/urdf/mahut_kol.urdf](assets/urdf/mahut_kol.urdf)):**
  * İlkel kutu/silindirler yerine gerçek `assets/meshes/*.stl` parçalarını ve paralel tutucuyu kullanan tam URDF dosyası oluşturuldu.
* **Yol Çözümleme ve Konfigürasyonlar:**
  * `Kol_model/meshes` sembolik bağı kurularak eski dosyalardaki bağıl yollar onarıldı.
  * `configs/robot/mahut_kol.yaml` ve `configs/task/pick_place.yaml` yeni eklem isimleri ve ortam koordinatlarına göre güncellendi.

---

## 7. 🦾 ManiSkill 3 Simülasyon Entegrasyonu ve Pick & Place Ortamı
* **URDF Dosya Yolu Uyumluluğu ([assets/urdf/mahut_kol.urdf](assets/urdf/mahut_kol.urdf)):**
  * SAPIEN 3 motorunun doğrudan yükleyebilmesi için ROS `package://mahut_kol/assets/meshes/` yolları `../meshes/` bağıl yollarına dönüştürüldü.
* **ManiSkill Robot Sınıfı Tamamlandı ([src/mahut_kol/robots/mahut_robot.py](src/mahut_kol/robots/mahut_robot.py)):**
  * `MahutRobot(BaseAgent)` sınıfı `@register_agent()` dekoratörüyle `"mahut_kol"` UID'si altında tescil edildi.
  * 4 kol eklemi (`joint1`..`joint4`) ve 2 tutucu parmak (`finger_left`, `finger_right`) için PD pozisyon ve delta kontrolcüleri (`PDJointPosControllerConfig`), keyframe pozisyonları ve çift parmak kavrama denetimi (`is_grasping`) implemente edildi.
* **Pick & Place Ortamları Kuruldu ([src/mahut_kol/envs/pick_place_task.py](src/mahut_kol/envs/pick_place_task.py)):**
  * `MahutKol-PickCube-v0`: Masa, kırmızı hedef küp, yeşil bırakma hedefi ve çoklu kameralar içeren bağımsız Pick & Place görevi kaydedildi.
  * `PickCubeMahutKol-v1`: ManiSkill 3 yerleşik masaüstü görevine `mahut_kol` robotu entegre edildi.
* **İnteraktif Simülasyon ve Görselleştirme Aracı ([scripts/sim_mahut_kol.py](scripts/sim_mahut_kol.py)):**
  * İster interaktif GUI penceresi (`--render-mode human`), ister video/resim kaydı (`--render-mode rgb_array`, `--save-video`, `--save-snapshot`) alabilen test betiği yazıldı.
  * Robotun masadaki küpe uzanıp kavrama sekansı başarıyla koşturulup `docs/mahut_kol_maniskill_sim.png` ve `docs/mahut_kol_reach.png` görselleri üretildi.
* **URDF Doğrulama Aracı Tamamlandı ([scripts/check_urdf.py](scripts/check_urdf.py)):**
  * SAPIEN 3 üzerinde link hiyerarşisini, aktif eklemleri ve limitleri doğrulayan test aracı hazırlandı.

---

## 8. 🎯 Özel Görev (Custom Task: PushCube) Modülü Oluşturuldu
* **ManiSkill 3 Özel Görev Sınıfı Eklendi ([src/mahut_kol/envs/push_cube_task.py](src/mahut_kol/envs/push_cube_task.py)):**
  * `MahutKolPushCubeEnv(BaseEnv)` sınıfı `@register_env("MahutKol-PushCube-v0")` ile tescillendi.
  * Masaya 3cm dinamik kırmızı küp (`cube`) ve kinematik hedef halkası (`goal_region`) yerleştirildi.
  * Epizot başında robot çalışma pozisyonuna ve küp hafif rassal koordinatlara konumlandırıldı.
  * End-effector, küp ve hedef pozisyonlarını kapsayan ek durum gözlemleri (`_get_obs_extra`, 29 boyutlu state) entegre edildi.
  * Başarı kriteri (`evaluate`) ve RL eğitimine uygun iki aşamalı ödül fonksiyonu (`compute_dense_reward`) yazıldı.
* **Görselleştirici ve Simülasyon Çalıştırıcı ([scripts/sim_custom_task.py](scripts/sim_custom_task.py)):**
  * PS4/Gamepad teleoperasyon, otomatik sinüs testi (`--mode sine`) ve anlık görsel alma (`--save-snapshot`) destekli betik eklendi.
  * Özel görevin simülasyon çıktısı [docs/mahut_kol_push_cube.png](docs/mahut_kol_push_cube.png) olarak belgelendi.
* **Birim Testler Genişletildi ([tests/test_custom_task.py](tests/test_custom_task.py)):**
  * `MahutKol-Minimal-v0` geriye dönük uyumluluk testi, `MahutKol-PushCube-v0` gözlem/aksiyon boyutu, ödül ve hedef başarı koşulunu doğrulayan testler yazıldı ve 9/9 başarıyla geçti.

---

## 9. 🚀 Yeni Robot Kol Modeli Entegrasyonu (5-DOF + Gripper)
* **Yeni CAD / MJCF Modelinin Sisteme Dahil Edilmesi (`assemble_new` Entegrasyonu):**
  * Robot koluna 100mm Lazy-Susan rulmanlı ve `joint0` taban dönüş motorlu (`base_rotate`) yeni gövde eklendi; kol 4-DOF'tan 5-DOF'a (taban yaw, omuz pitch, dirsek pitch, bilek pitch, bilek roll) ve 2 parmak paralel kıskaç ile toplam 7 aktif ekleme yükseltildi.
  * 5 adet yeni taban mesh dosyası (`base_rotate-v1_*.stl`) `assets/meshes/` dizinine aktarıldı.
  * Bağımsız robot modeli (`assets/mahut_kol_robot.xml`) ve ManiSkill için optimize edilmiş MJCF (`assets/mahut_kol_maniskill.xml`) güncellendi.
  * Çalışma masası, kırmızı küp ve hedef alanını içeren simülasyon sahnesi (`assets/mahut_kol_scene.xml`) yeni robot kinematiğine uyarlandı.
* **Matematiksel Birebir URDF Üretimi ([assets/urdf/mahut_kol.urdf](assets/urdf/mahut_kol.urdf)):**
  * `scripts/generate_exact_urdf.py` aracı yenilendi; MuJoCo modelinden gerçek kütle, kütle merkezi ve atalet tensörlerini (`inertial`) ve eklem sınırlarını çekerek SAPIEN uyumlu 10 link ve 7 aktif eklemli URDF üretildi ve `scripts/check_urdf.py` ile doğrulandı.
* **Robot ve Görev Sınıflarının Güncellenmesi:**
  * [configs/robot/mahut_kol.yaml](configs/robot/mahut_kol.yaml): 5 kol eklemi ve kıskaç için güncel tork limitleri, servo PID kazançları ve başlangıç pozisyonu yapılandırıldı.
  * [src/mahut_kol/robots/mahut_robot.py](src/mahut_kol/robots/mahut_robot.py): `MahutRobot` sınıfı 7 eklemli yeni modele uyarlandı.
  * [src/mahut_kol/envs/minimal_task.py](src/mahut_kol/envs/minimal_task.py) ve [push_cube_task.py](src/mahut_kol/envs/push_cube_task.py): Başlangıç pozisyonları ve durum gözlem boyutları (31-dim) güncellendi.
  * [src/mahut_kol/envs/pick_cube_task.py](src/mahut_kol/envs/pick_cube_task.py): Ters kinematik (IK/FK) hesaplayıcı ve otonom gösterim (scripted policy) içeren Pick & Place ortamı (`MahutKol-PickCube-v1`) eklendi.
* **Teleoperasyon ve Araçların Genişletilmesi:**
  * [src/mahut_kol/utils/teleop_ps4.py](src/mahut_kol/utils/teleop_ps4.py): 7 boyutlu aksiyon haritalaması yapıldı (Sol stick: `joint0`/`joint1`, Sağ stick: `joint2`/`joint3`, D-Pad/Droll: `joint4`, Tetikler: Kıskaç).
  * Yeni analiz ve eğitim araçları eklendi: `scripts/test_robot.py` (eklem limit testi), `scripts/load_test.py` (statik taşıma kapasitesi analizi), `scripts/collect_demos.py` (LeRobot veri kümesi toplayıcı), `scripts/train_smolvla.sh` (SmolVLA eğitim başlatıcı).
* **Test Doğrulaması:**
  * [tests/test_custom_task.py](tests/test_custom_task.py) ve [tests/test_teleop.py](tests/test_teleop.py) güncellendi; 11 birim testin tamamı (11/11) hatasız geçti.

---

## 10. ⚡ Donanım Spesifikasyonu (FT5330M & ESP32-S3) ve Fizik Motoru Dengeleme
* **Fiziksel Donanım Dokümantasyonu Eklendi ([docs/DONANIM_BILGILERI.md](docs/DONANIM_BILGILERI.md)):**
  * **Feetech FT5330M 35Kg (7.4V)** motor verileri (stall torku: 35 kg·cm = 3.432 Nm, stall akımı: ~3.5A, gerilim sınırları) belgelendi.
  * **ESP32-S3-N16R8** kontrolcü kartı (16MB Flash, 8MB PSRAM, MCPWM servo kontrolü, Wi-Fi/BLE) entegrasyonu açıklandı.
  * Kramayer-pinyon mekaniği ($m=0.8$, $z=18$, $r=7.2\text{mm}$) üzerinden teorik 477 N kuvvet ile fiziksel emniyetli kavrama kuvveti (15 N) arasındaki Sim-to-Real ilişkisi belgelendi.
* **Kıskaç Kontrolcü Parametreleri Dengeledi ([src/mahut_kol/robots/mahut_robot.py](src/mahut_kol/robots/mahut_robot.py)):**
  * Kol eklemlerinde FT5330M'nin gerçek stall torku (`3.432 Nm`) korunarak kıskaç kuvvet limiti `477.0 N`'dan gerçekçi ve emniyetli `15.0 N`'a, sertlik `2000.0`'dan `150.0`'a, sönümleme `15.0`'a çekildi.
  * Kıskaç kapandığında veya küp sıkıştırıldığında oluşan aşırı reaksiyon kuvvetlerinin kolu geriye savurması engellendi.
* **URDF İç Çarpışmaları Temizlendi ([assets/urdf/mahut_kol.urdf](assets/urdf/mahut_kol.urdf)):**
  * `gripper_gear_link` içindeki hatalı `<collision>` kaldırıldı; parmakların iç dişli parçasına çarparak `joint4` bilek dönüşünü kendi kendine döndürmesi (%90+ sarsıntı) tamamen giderildi.
  * `scripts/generate_exact_urdf.py` güncellenerek gelecekteki üretimlerde dahili dişli ve mekanizma parçalarının collision alması engellendi.
* **Scripted Actions ve Veri Toplama Boyut Uyumluluğu:**
  * [src/mahut_kol/envs/pick_cube_task.py](src/mahut_kol/envs/pick_cube_task.py) içindeki `scripted_actions` 7 eklemli modele tam uyarlanarak 7-boyutlu (`np.r_[q, 0, finger, finger]`) eylem üretir hale getirildi; `collect_demos.py` güncellendi.
* **Doğrulama:**
  * Tüm birim testler (11/11) ve kıskaç dinamik kararlılık testleri başarıyla doğrulandı.

---

## 11. 🛡️ Küp Tutma Sırasında Kolun Kayması (Drifting) ve Küpün Düşmesi Sorununun Çözümü
* **Kök Neden Tespiti (Root Cause Analysis):**
  1. **`pd_joint_delta_pos` Kontrolcüsünün Hedef Hafızası Olmaması (`use_target=False`):** ManiSkill'in bu modunda `target_qpos = current_qpos + action` formülü çalışır. Tetik veya analog çubuk bırakıldığında (`action = 0`), hedef pozisyon anlık mevcut açıya eşitlenir ($e = \text{target} - \text{current} = 0$). Pozisyon hatası sıfırlandığı için PD motor torku **0 Nm**'ye düşer; kıskaç küpü sıkıştırmayı anında bırakır ve küp yerçekimiyle düşer. Kol da yerçekimi/temas itmesiyle her adımda itildiği açıyı yeni hedef kabul ederek yana/aşağı doğru pasif olarak kayar (passive drifting).
  2. **Eklem Hedef Aşımı (Target Wind-Up):** Tetik basılı tutulduğunda `_target_qpos` eklem limitlerini aşarak $-10\text{ rad}$ veya $+12\text{ rad}$ gibi aşırı değerlere fırlamakta, bu da PhysX temas çözücüsünde binlerce Newtonluk temas reaksiyon kuvveti oluşturup kolu masada savurmaktaydı.
  3. **Küpün Düşük Sürtünmesi:** `actors.build_cube` varsayılan sürtünmesi ($0.3$) ıslak teflon gibi davranmaktaydı; katı parmaklar en ufak titreşimde küpü kaydırmaktaydı.
  4. **Z-Kot Masa Penetrasyonu:** Robot masanın $3.6\text{ cm}$ içine gömülerek (`z = 0`) başlatıldığı için gövde masayla temas halinde reaksiyon kuvveti üretiyordu.
* **Uygulanan Çözümler:**
  * **`pd_joint_target_delta_pos` Moduna Geçildi:** `use_target = True` yapıldı. Artık tetik veya analog bırakıldığında (`action = 0`), hedef pozisyon kilitli kalır ($\Delta q = 0$). Kontrolcü tam torkla küpü sıkıştırmaya ve kolu havada sabit tutmaya devam eder.
  * **Target Clamping Eklendi:** Her simülasyon adımında `controller._target_qpos` değeri robotun gerçek eklem limitlerine (`agent.robot.get_qlimits()`) sınırlandırıldı (clamp). Wind-up patlaması ve geri dönüş gecikmesi tamamen engellendi.
  * **Küp Sürtünmesi Arttırıldı:** Küp temas geometrisine silikon/kauçuk kavrama sürtünmesi ($\mu_{\text{static}} = 2.0, \mu_{\text{dynamic}} = 2.0$) atandı.
  * **Z-Kot ve Self-Collision Düzeltildi:** Taban yüksekliği `sapien.Pose(p=[0, 0, 0.0558])` olarak düzeltildi ve `MahutRobot.disable_self_collisions = True` yapıldı.
  * **Teleoperasyon Betiği ([scripts/teleop_maniskill_ps4.py](scripts/teleop_maniskill_ps4.py)):** Hem MahutKol (7 DoF) hem SO-100 (6 DoF) için SAPIEN 3 `set_drive_properties` motor tork güvenliği ve klavye/gamepad eşleşmeleriyle yeniden yapılandırıldı.
* **Doğrulama Sonuçları:**
  * **MahutKol Kayma Testi:** 50 adım boyunca havada sıfır eylemle tutulduğunda eklem kayması: **`0.00000589 rad`** (neredeyse sıfır).
  * **SO-100 Kayma Testi:** 50 adım boyunca havada tutulduğunda kayma: **`0.000001 rad`**.
  * **Birim Testler:** 11/11 birim test başarıyla geçti.

---

## 12. 🔄 Model Yapılandırmasının Özel Modele (MahutKol 5-DOF + Kıskaç) Tam Geçirilmesi
* **Kök Neden:**
  * Kod tabanında ve teleoperasyon betiğinde (`scripts/teleop_maniskill_ps4.py`) varsayılan ortam olarak SO-100 robotu (`PickCubeSO100-v1`) kalmıştı; ayrıca proje rehberinde robot modeli hatalı olarak SO-100 olarak anılmaktaydı.
* **Yapılan Düzeltmeler:**
  1. **Teleoperasyon Betiği ([scripts/teleop_maniskill_ps4.py](scripts/teleop_maniskill_ps4.py)):**
     * Varsayılan ortam `--env-id` değeri doğrudan kullanıcının özel robot modeli olan `MahutKol-PickCube-v1` yapıldı.
     * Konsol raporları ve kontroller 7-DOF MahutKol (`joint0`..`joint4` + `finger_left`/`finger_right`) modelini birincil kabul edecek şekilde güncellendi.
  2. **Veri Toplama ve Değerlendirme Betikleri:**
     * `src/mahut_kol/teleop/record_demos.py` ve `src/mahut_kol/eval/evaluate_policy.py` içerisindeki varsayılan ortam `MahutKol-PickCube-v1` olarak güncellendi.
     * `configs/task/pick_place.yaml` içindeki `env_id` değeri `MahutKol-PickCube-v1` ile senkronize edildi.
  3. **Dokümantasyon Temizliği ([PROJE_REHBERI.md](PROJE_REHBERI.md)):**
     * SO-100 tabanlı eski ifadeler kaldırıldı; projenin kullanıcının kendi özel 5-DOF kol + 2-parmak paralel kıskaçlı modeli, FT5330M servo motorları ve ESP32-S3 mimarisi üzerinde çalıştığı netleştirildi.
     * Proje ağacı ve hızlı başlangıç komutları güncel MahutKol araçlarıyla (`teleop_maniskill_ps4.py`, `sim_custom_task.py`, `view_mujoco.py`, `collect_demos.py`, `train_smolvla.sh`) güncellendi.
  4. **Doğrulama:**
     * Tüm birim testler (11/11) ve ortam başlatma testleri başarıyla doğrulandı.

---

## 13. 🎮 Teleoperasyon Kodlarının `Teleoperasyonlar/` Klasöründe Toplanması
* **Yapılan İşlemler:**
  1. **Yeni Klasör:** Proje kök dizininde `Teleoperasyonlar/` klasörü açıldı.
  2. **Taşınan ve Merkezi Hale Getirilen Kodlar:**
     * `Teleoperasyonlar/teleop_maniskill_ps4.py`: PS4 / Gamepad ve klavye ile canlı simülasyon kontrolü sağlayan ana teleoperasyon betiği.
     * `Teleoperasyonlar/test_ps4.py`: Gamepad eksen ve buton donanım test aracı.
     * `Teleoperasyonlar/record_demos.py`: Teleoperasyonla insan gösterimi (demonstration) toplayıp HDF5 formatına kaydedici.
     * `Teleoperasyonlar/teleop_agent.py`: Pygame tabanlı Kartezyen `TeleopController` sınıfı.
     * `Teleoperasyonlar/teleop_ps4.py`: 7-DoF eklem uzayı delta pozisyon `MahutPS4Controller` sınıfı.
     * `Teleoperasyonlar/__init__.py`: Python paketi olarak dışa aktarım desteği.
     * `Teleoperasyonlar/README.md`: Hızlı başlangıç komutları ve PS4/klavye tuş haritası dokümantasyonu.
  3. **Geriye Dönük Uyumluluk:**
     * `scripts/teleop_maniskill_ps4.py` ve `scripts/test_ps4.py` yönlendirici (forwarder) betikler haline getirildi.
     * `src/mahut_kol/utils/teleop_ps4.py` ve `src/mahut_kol/teleop/` modülleri `Teleoperasyonlar/` paketini dışa aktaracak şekilde köprülendi; mevcut testlerin ve importların kesintisiz çalışması sağlandı.

---

## 14. 🎙️ Yerel (Offline) Türkçe Speech-to-Text ve Robot Komut Çözücü Modülü
* **Geliştirilen Mimari ve Bileşenler (`src/mahut_kol/speech/`):**
  1. **Komut Çözücü ve Normalizasyon ([command_parser.py](src/mahut_kol/speech/command_parser.py)):**
     - Türkçe büyük/küçük harf (`İ`/`i`, `I`/`ı`) normalizasyonu (`turkish_lower`).
     - Morfolojik ek çözümleme (akkusatif, yönelme, tamlayan ekleri: "küpü", "silindirin", "prizmaya", "sağına", "önüne").
     - Renkler (`red`, `green`, `blue`), şekiller (`cube`, `cylinder`, `triangular_prism`), uzamsal yönler (`FRONT`, `BACK`, `LEFT`, `RIGHT`, `TOP`) ve eylemler (`pick_and_place`, `pick`, `push`) için deterministik slot extraction kural tabanı.
     - `multi_shape_task.py` ortamının nesne kimlikleri ve hedef slot formatlarıyla (`green_cylinder_RIGHT` vb.) tam uyumlu JSON çıktısı.
  2. **Ses Kayıt Modülü ([audio_recorder.py](src/mahut_kol/speech/audio_recorder.py)):**
     - 16 kHz Mono ses standardı (`sounddevice` + `numpy`).
     - Hem Bas-Konuş (Push-to-Talk) hem de Enerji Tabanlı Sessizlik Tespiti (VAD) kayıt modları.
     - Root iznine gerek kalmadan `~/.local/lib/libportaudio.so.2` kütüphanesini otomatik keşfeden PortAudio dinamik bağlama kancası.
  3. **STT Motoru ([stt_engine.py](src/mahut_kol/speech/stt_engine.py)):**
     - `faster-whisper` CTranslate2 model sarmalayıcısı (`language="tr"`).
     - Otomatik donanım seçimi: RTX 4060 GPU üzerinde CUDA `float16`, arıza/OOM durumunda otomatik CPU `int8` fallback desteği.
  4. **Orkestrasyon Sınıfı ([voice_commander.py](src/mahut_kol/speech/voice_commander.py)):**
     - Ses yakalama, transkripsiyon ve komut çözümlemeyi tek bir uçtan uca arayüzde birleştiren ana sınıf (`listen_and_parse`, `process_text`, `process_audio_file`, `run_interactive_loop`).
  5. **Canlı Doğrulama Aracı ([scripts/test_voice_command.py](scripts/test_voice_command.py)):**
     - CLI üzerinden `--mode text`, `--mode push_to_talk`, `--mode vad` ve `--mode file` seçenekleriyle renkli formatta canlı test aracı.
* **Test ve Doğrulama:**
  - 4 yeni birim ve entegrasyon test modülü yazıldı (`test_command_parser.py`, `test_audio_recorder.py`, `test_stt_engine.py`, `test_voice_commander.py`).
  - Projedeki toplam 71 birim testin tamamı (`python3 -m unittest discover -s tests`) hatasız geçti.
