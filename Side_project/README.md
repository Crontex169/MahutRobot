# SO-100 (SO-ARM100) MuJoCo Simülasyonu

Bu klasör, **The Robot Studio** ve **Hugging Face LeRobot** tarafından geliştirilen açık kaynaklı **SO-100 (SO-ARM100)** 6 serbestlik dereceli (DoF) robot kolunun MuJoCo simülasyon modelini ve görselleştirme betiğini içerir. Model dosyaları Google DeepMind'ın resmi [MuJoCo Menagerie (`trs_so_arm100`)](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100) deposundan temin edilmiştir.

## İçerik

- `view_so100.py` / `main.py`: MuJoCo görüntüleyiciyi (viewer) başlatan Python betiği.
- `so_arm100/`:
  - `scene.xml`: Zemin düzlemi, gökyüzü (skybox) ve ışıklandırma içeren sahne dosyası.
  - `so_arm100.xml`: SO-100 robotunun kinematiği, eklemleri, dinamikleri ve çarpışma geometrilerini içeren MJCF dosyası.
  - `assets/`: 3D STL mesh dosyaları (Base, Upper Arm, Lower Arm, Jaw vb.).

## Gereksinimler

MuJoCo Python kütüphanesi:
```bash
pip install mujoco
```

## Çalıştırma

Modeli görselleştirmek için:

```bash
# Varsayılan sahne ile interaktif pencereyi açar:
python3 Side_project/view_so100.py

# veya Side_project dizini içindeyken:
python3 view_so100.py
# veya
python3 main.py
```

### Seçenekler

- **Sadece robot modeli (zemin ve gökyüzü olmadan):**
  ```bash
  python3 view_so100.py --model-only
  ```

- **Farklı bir başlangıç pozu ile başlatma (`home` veya `rest`):**
  ```bash
  python3 view_so100.py --keyframe rest
  ```

- **Pasif görselleştirici modu:**
  ```bash
  python3 view_so100.py --passive
  ```

### Fare ve Klavye Kontrolleri

| Girdi | Eylem |
|---|---|
| **Sol Tık + Sürükle** | Kamerayı döndürür |
| **Sağ Tık + Sürükle** | Yakınlaştırır / Uzaklaştırır |
| **Orta Tık + Sürükle** | Kamerayı kaydırır (Pan) |
| **Çift Tık** | Seçilen gövdeye/ekleme odaklanır |
| **Ctrl + Sağ Tık Sürükle** | Robota etkileşimli kuvvet uygular |
| **Boşluk (Space)** | Simülasyonu duraklatır / devam ettirir |
| **Backspace** | Simülasyonu sıfırlar |
