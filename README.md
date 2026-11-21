# MahutKol

Özel 5-DOF robot kol + paralel tutucu projesi. Depo dört ana hattan oluşur:

| Klasör | Hat |
|---|---|
| `Main_project/` | Ana hat: ManiSkill simülasyonu, teleop demo kaydı, LeRobot veri kümesi, VLA eğitimi (Modal) ve robotumuza göre ince ayar, kapalı çevrim değerlendirme. Python paketi `mahut_kol` burada. |
| `Real-Sim/` | Gerçek kol ↔ simülasyon/bilgisayar köprüsü: ESP32-S3 firmware, LeRobot robot sürücüsü (`lerobot_robot_mahut`). |
| `Paper/` | Türkçe dil talimatlarıyla LIBERO üzerinde sonuçlar (makale çalışması). |
| `Side_project/` | Yalnızca MuJoCo teleoperasyon denemeleri (SO-100, WebXR/webcam); ana pakete bağlı değil. |

Ortak: `Fikirler/` (PM notları ve promptlar), `graphify-out/` (otomatik üretilen bilgi grafı).

Ana hat kurulumu:

```bash
cd Main_project && pip install -r requirements.txt && pip install -e .
```

Ayrıntılar: `Main_project/README.md`, `Main_project/PROJE_REHBERI.md`.
