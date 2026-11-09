# 🔄 MahutKol: ManiSkill & LeRobot Entegrasyon Hattı

Bu doküman, **ManiSkill 3** simülasyon ortamı ile **Hugging Face LeRobot** eğitim çerçevesinin projemizde nasıl birlikte çalıştığını ve veri akışının adımlarını açıklar.

---

## 🏗️ Uçtan Uca Mimari ve Veri Akışı

```text
┌─────────────────────────┐
│ CAD Tasarımı (STL/STP)   │ ──▶ assets/meshes/
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ URDF + Meshler           │ ──▶ assets/urdf/mahut_kol.urdf
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ ManiSkill Robot & Görev │ ──▶ src/mahut_kol/robots/mahut_robot.py (+ kinematics.py: URDF FK/IK)
│ Simülasyon Ortamı       │ ──▶ src/mahut_kol/envs/multi_shape_task.py (MahutKol-MultiShapeRelational-v1)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Komut Katmanı           │ ──▶ src/mahut_kol/commands/ (CommandParser JSON → reset(options),
│                         │     İngilizce talimat, slotlar, dengeli kuyruk, held-out)
└────────────┬────────────┘
             ▼ (telefon WebXR / PS4 / klavye  |  scripted politika)
┌─────────────────────────┐
│ Gösterim Kaydı          │ ──▶ scripts/record_vla_demos.py (teleop)
│ (doğrudan LeRobot v3)   │ ──▶ scripts/generate_scripted_demos.py (scripted)
│                         │     src/mahut_kol/recording/ (yazıcı, sidecar, kayıt döngüsü)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Doğrulama + Kapsama      │ ──▶ scripts/validate_dataset.py
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Model Eğitimi           │ ──▶ cloud/mahut_cloud.py (LeRobot; Modal, bkz. PROJE_REHBERI §9)
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ Kapalı Çevrim Test      │ ──▶ scripts/eval_policy.py (nesne/yön, held-out, dil duyarlılığı)
└─────────────────────────┘
```

Eski HDF5 → LeRobot dönüştürme adımı ve `src/mahut_kol/datasets/`, `train/`, `eval/` stub
paketleri kaldırıldı: kayıt betikleri veriyi doğrudan `LeRobotDataset` (lerobot 0.4.4,
codebase v3.0) formatında yazar. Yeni kodu `datasets/` adlı klasöre koymayın (.gitignore).

---

## 📂 Kod Dosyaları ve Kütüphane Rolleri

| Dosya Yolu | Sorumlu Kütüphane | Rol ve İşlev |
| :--- | :--- | :--- |
| `src/mahut_kol/robots/mahut_robot.py` | **ManiSkill 3** | Robotu `BaseAgent` olarak tanımlar, eklemleri eşleştirir, PD kazançlarını belirler. |
| `src/mahut_kol/robots/kinematics.py` | **numpy** | URDF'ten FK/Jacobian/eğim kısıtlı DLS IK; parmak-masa çarpışma kontrolü. |
| `src/mahut_kol/envs/multi_shape_task.py` | **ManiSkill 3** | 9 nesneli ilişkisel yerleştirme; çalışma alanı, yön tanımları, başarı (yerleşti + bırakıldı + durağan), front/wrist kameraları. |
| `src/mahut_kol/envs/scripted_policy.py` | **numpy** | Otomatik gösterim politikası (yaklaş-kavra-kaldır-taşı-bırak). |
| `src/mahut_kol/commands/` | — | Komut şeması, İngilizce şablonlar, slotlar, dengeli kuyruk, held-out, komut kaynakları (queue/text/voice). |
| `src/mahut_kol/teleop/phone_webxr.py` | **FastAPI + uvicorn** | Telefon WebXR teleopu (WebSocket, debriyaj, EMA, IK) → `pd_joint_pos`. |
| `src/mahut_kol/teleop/action_source.py` | — | Cihazdan bağımsız `ActionSource` (telefon, PS4/klavye, scripted). |
| `src/mahut_kol/recording/` | **LeRobot** | Veri kümesi yazıcı (oluştur/devam et/ekle, şema koruması), sidecar JSONL, kayıt döngüsü. |
| `scripts/validate_dataset.py` | **LeRobot + pandas** | Şema, kare sayıları, NaN, video, sidecar, kapsama, held-out sızıntısı. |
| `scripts/eval_policy.py` | **ManiSkill + LeRobot** | Eğitilmiş politikayı nesne/yön, held-out ve dil duyarlılığı açısından ölçer. |

---

## 🔗 İlgili Ayrıntılı Kılavuzlar
* [ManiSkill 3 Kılavuzu](MANISKILL_GUIDE.md)
* [LeRobot Kılavuzu](LEROBOT_GUIDE.md)
* [VLA Görev Tanımları](VLA_GOREV_TANIMLARI.md)
