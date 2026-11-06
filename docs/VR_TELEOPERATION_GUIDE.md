# 🥽 VR (Sanal Gerçeklik) ile Teleoperasyon Kılavuzu

Evet, hem **ManiSkill 3** hem de **LeRobot** ekosistemlerinde **VR (özellikle Meta Quest 2 / 3 / Pro)** başlıkları ile 6-DoF teleoperasyon ve yüksek kaliteli insan gösterimi (demonstration) toplama desteği bulunmaktadır.

Bu kılavuz, VR teleoperasyonun nasıl çalıştığını, mimarisini ve `MahutKol` projesine nasıl entegre edileceğini açıklar.

---

## 1. Neden VR Teleoperasyon?

| Kontrol Yöntemi | Serbestlik (DoF) | Sezgisellik | Veri Toplama Hızı & Kalitesi |
| :--- | :--- | :--- | :--- |
| **Klavye / Mouse** | 1-2 eksen aynı anda | Düşük (Kademeli butonlar) | Yavaş, yapay/kesintili yörüngeler |
| **Gamepad (Xbox vb.)** | 2-4 eksen (Thumbsticks) | Orta | Orta seviye, çok eklemli işlerde zor |
| **SpaceMouse** | 6-DoF Kartezyen | İyi | Akıcı ancak derinlik algısı ekranda sınırlı |
| **VR (Meta Quest vb.)** | **6-DoF El / Kontrolcü** | **Mükemmel (Doğal insan eli)** | **En yüksek (Hızlı, 3D derinlik algılı, akıcı)** |

VR sayesinde operatör 3 boyutlu ortamı stereoskopik derinlikle görür, kontrolcünün uzaydaki pozisyonu ve rotasyonu (6-DoF pose) anlık olarak robotun uç organına (End-Effector) haritalanır ve tetik tuşu (trigger) tutucuyu (gripper) açıp kapatır.

---

## 2. ManiSkill 3'te VR Teleoperasyon

ManiSkill 3, robot manipülasyonu için yerleşik VR teleoperasyon araçlarına sahiptir:

### A. Mimari & Veri İletimi
1. **Görüntü Akışı (Visual Streaming):**
   * SAPIEN simülasyon sahnesi stereo (çift kamera) olarak render edilir.
   * **ALVR (Air Light VR)** veya **WebXR** üzerinden Meta Quest başlığına düşük gecikmeli (low-latency, 60 Hz / 4K) kablosuz video akışı sağlanır.
2. **Girdi Eşleştirme (Input Tracking):**
   * Meta Quest kontrolcüsünün uzaysal pozisyonu `[x, y, z]` ve oryantasyonu `[qx, qy, qz, qw]` anlık okunur.
   * ManiSkill'in **PDEEPoseController** veya yerleşik **Ters Kinematik (IK - Pinocchio)** çözücüsü, bu pozu robotun eklem açılarına (`qpos`) dönüştürür.

### B. ManiSkill Yerleşik VR Betikleri
ManiSkill deposunda yer alan teleoperasyon referansları:
* `mani_skill/examples/teleoperation/` altındaki VR akış betikleri (`demo_vr_teleop_stream.py`).
* WebXR tabanlı `maniskill-quest-teleop` veya `ImVR` eklentileri.

---

## 3. LeRobot Ekosisteminde VR Desteği

LeRobot topluluğu ve eklenti mimarisi, Meta Quest ile doğrudan LeRobot veri setleri toplamayı destekler:

1. **WebXR / Vuer Tabanlı Eklentiler (`lerobot_teleoperator_vuer`):**
   * Meta Quest'in dahili tarayıcısı (Oculus Browser) açılır.
   * Bilgisayarda çalışan Vuer / WebXR sunucusuna bağlanılır.
   * Herhangi bir APK yüklemeden doğrudan 6-DoF el takibi ve kontrolcü verileri LeRobot döngüsüne akar.
2. **phosphobot & BEAVR:**
   * Quest başlığı için özel geliştirilmiş arayüzler. Robot kolunu sürerken toplanan yörüngeleri doğrudan Hugging Face `LeRobotDataset` formatında diske kaydeder.

---

## 4. `MahutKol` Pipeline'ına VR Entegrasyonu

Projemizdeki [teleop_agent.py](file:///home/deniz/Ortak/MahutKol/src/mahut_kol/teleop/teleop_agent.py) ve [record_demos.py](file:///home/deniz/Ortak/MahutKol/src/mahut_kol/teleop/record_demos.py) dosyalarına VR desteği şu şekilde eklenir:

### Adım 1: Eylem Kontrolcüsünü Kartezyen EEF Moda Alma
VR kullanırken robot eklemleri tek tek değil, End-Effector hedef pozu olarak kontrol edilir:
```python
# configs/robot/mahut_kol.yaml
controllers:
  arm:
    controller_type: "pd_ee_delta_pose"  # [dx, dy, dz, droll, dpitch, dyaw]
```

### Adım 2: TeleopController İçine VR Arayüzü Ekleme
```python
# src/mahut_kol/teleop/teleop_agent.py
class TeleopController:
    def __init__(self, device_type: str = "keyboard"):
        self.device_type = device_type
        if device_type == "vr":
            # WebXR veya OpenXR / ALVR istemcisini başlat
            self.init_vr_interface()

    def get_action(self) -> np.ndarray:
        if self.device_type == "vr":
            # VR kontrolcüsünden delta pose ve trigger durumunu oku
            delta_xyz, delta_rot, gripper_val = self.read_vr_hand_pose()
            return np.concatenate([delta_xyz, delta_rot, [gripper_val]])
```

---

## 5. Gereksinimler & Kurulum Adımları (Meta Quest)

* **Donanım:** Meta Quest 2, Quest 3 veya Quest Pro.
* **Bağlantı Seçenekleri:**
  1. **Wi-Fi (Kablosuz):** 5 GHz / Wi-Fi 6 yerel ağ üzerinde ALVR veya AirLink.
  2. **Link Kablosu (Kablolu):** USB-C 3.0 Oculus Link kablosu (en düşük gecikme).
* **Yazılımlar:**
  * SteamVR + ALVR (Linux / Ubuntu üzerinde Quest akışı için en popüler açık kaynak çözüm).
  * WebXR (Ekstra yazılım kurmadan tarayıcı üzerinden hızlı prototipleme).
