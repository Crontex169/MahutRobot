# Fizik Motoru Dengeleme ve Donanım Entegrasyonu Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feetech FT5330M (35 kg·cm) motor ve ESP32-S3-N16R8 kontrolcü donanım spesifikasyonlarını belgelemek, kıskaç kuvvet limitlerini ve sertliklerini gerçekçi değerlere çekmek, URDF iç çarpışmalarını temizlemek ve robot kolun küpü tutarken savrulmasını engellemek.

**Architecture:** Fizik motorundaki kararsızlıkların (477N aşırı kuvvet, iç parça temasları, 6/7 aksiyon boyutu uyuşmazlığı) kök nedenlerini gidermek; simülasyon kontrol parametrelerini fiziksel donanımın sınırlarına (Sim-to-Real) uygun hale getirmek.

**Tech Stack:** Python 3.10, ManiSkill 3 / SAPIEN 3, MuJoCo 3.15, Pygame, URDF/MJCF.

**Spec:** Feetech FT5330M (7.4V, 35 kg·cm stall torku = 3.432 Nm), ESP32-S3-N16R8 (16MB Flash, 8MB PSRAM), Gripper-v2 kramayer mekaniği.

## Global Constraints
- Feetech FT5330M kol eklem tork limiti olan `3.432 Nm` (`joint0`..`joint4`) korunmalıdır.
- Kıskaç parmak kuvvet limiti fiziksel tutma ve ESP32 akım limitlerine uygun olarak `15.0 N`, sertliği `150.0` olarak ayarlanmalıdır.
- Tüm birim testler (`python3 -m unittest discover tests`) hatasız geçmelidir.

## Review Focus
- Kıskaç kapandığında `joint4` veya kol eklemlerinde istenmeyen reaksiyonel açı sapması oluşmamalıdır.
- `gripper_gear_link` ile parmaklar arasında self-collision olmamalıdır.
- `pick_cube_task.py` içindeki `scripted_actions` 7 boyutlu eylem üretmelidir.

---

### Task 1: Donanım Spesifikasyonu Dokümantasyonu

**Files:**
- Create: `docs/DONANIM_BILGILERI.md`
- Modify: `docs/README.md`
- Modify: `PROJE_REHBERI.md`

- [ ] **Step 1: `docs/DONANIM_BILGILERI.md` dosyasını oluştur**
Feetech FT5330M (35 kg·cm, 3.432 Nm, 7.4V), ESP32-S3-N16R8, pinyon/kramayer dişli hesabı, Sim-to-Real yaklaşımı ve kuvvet parametrelerinin açıklamalarını yaz.

- [ ] **Step 2: `docs/README.md` ve `PROJE_REHBERI.md` içine donanım kılavuzunu ekle**
Dokümantasyon merkezine `DONANIM_BILGILERI.md` bağlantısını ekle.

- [ ] **Step 3: Doğrula**
Dosyaların eksiksiz oluşturulduğunu kontrol et.

---

### Task 2: Robot Kontrolcü ve Konfigürasyon Parametrelerinin Düzeltilmesi

**Files:**
- Modify: `src/mahut_kol/robots/mahut_robot.py`
- Modify: `configs/robot/mahut_kol.yaml`

- [ ] **Step 1: `src/mahut_kol/robots/mahut_robot.py` kontrolcü kazançlarını güncelle**
`stiffness = [80.0, 80.0, 80.0, 80.0, 80.0, 150.0, 150.0]`
`damping = [10.0, 10.0, 10.0, 10.0, 10.0, 15.0, 15.0]`
`force_limit = [3.432, 3.432, 3.432, 3.432, 3.432, 15.0, 15.0]`

- [ ] **Step 2: `configs/robot/mahut_kol.yaml` dosyasını güncelle**
Gripper için `stiffness: 150.0`, `damping: 15.0`, `force_limit: 15.0` olarak güncelle.

---

### Task 3: URDF İç Çarpışmalarının ve Scripted Actions Boyutunun Düzeltilmesi

**Files:**
- Modify: `assets/urdf/mahut_kol.urdf`
- Modify: `scripts/generate_exact_urdf.py`
- Modify: `src/mahut_kol/envs/pick_cube_task.py`

- [ ] **Step 1: `assets/urdf/mahut_kol.urdf` içindeki `gripper_gear_link` collision bloğunu kaldır**
İç dişli parçasının parmaklarla çarpışarak tork üretmesini engelle.

- [ ] **Step 2: `scripts/generate_exact_urdf.py` içinde dahili parça filtrelemesini güçlendir**
Gelecekte URDF yeniden üretildiğinde `part-13-v1` veya iç mekanizma parçalarının collision almasını önle.

- [ ] **Step 3: `src/mahut_kol/envs/pick_cube_task.py` içinde `scripted_actions` fonksiyonunu 7-boyutlu yap**
`np.r_[q, 0, finger, finger]` eylemini yield ederek 7 eklemli robot ile tam uyumlu hale getir.

---

### Task 4: Birim Testlerin ve Simülasyon Doğrulamasının Çalıştırılması

**Files:**
- Test: `tests/test_custom_task.py`
- Test: `tests/test_teleop.py`
- Modify: `YAPILANLAR.md`

- [ ] **Step 1: Test paketini çalıştır**
`python3 -m unittest discover tests`

- [ ] **Step 2: Kıskaç kapatma simülasyon testi yap**
Kıskaç kapatıldığında `joint4` ve kol eklemlerinin savrulmadığını ve stabil kaldığını doğrula.

- [ ] **Step 3: `YAPILANLAR.md` dosyasını güncelle**
Yapılan donanım kaydı ve dinamik kararlılık düzeltmelerini günlüğe ekle.
