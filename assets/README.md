# CAD ➔ URDF Dönüştürme ve Hazırlık Kılavuzu

Kendi tasarladığınız robotik kolu **ManiSkill** (SAPIEN / PhysX) ortamında stabil ve hatasız çalıştırmak için aşağıdaki adımları ve kuralları izleyin.

---

## 1. Klasör Düzeni

```text
assets/
├── urdf/
│   └── mahut_kol.urdf            # Ana URDF dosyanız
└── meshes/
    ├── visual/                   # Görsel amaçlı yüksek poligonlu mesh'ler (OBJ/DAE/STL)
    │   ├── base_link.obj
    │   ├── link1.obj
    │   └── ...
    └── collision/                # Fizik motoru için sadeleştirilmiş konveks mesh'ler
        ├── base_link_collision.obj
        ├── link1_collision.obj
        └── ...
```

---

## 2. Altın Kurallar (Simülasyon Hatalarını Önleme)

### A. Görsel vs. Çarpışma Mesh'leri (Visual vs Collision)
- **Visual Mesh:** CAD'den çıktığı gibi detaylı olabilir. OBJ veya DAE formatı renk ve doku için önerilir.
- **Collision Mesh:** **ASLA** doğrudan yüksek poligonlu ham CAD mesh'ini collision olarak tanımlamayın! PhysX çöker veya simülasyon aşırı yavaşlar.
- **CoACD ile Konveks Ayrıştırma (Tavsiye):**
  ```bash
  # CoACD CLI ile collision mesh üretme
  coacd -i meshes/visual/link1.obj -o meshes/collision/link1_collision.obj -t 0.05
  ```

### B. Atalet Matrisleri (Inertia Moments)
- URDF içindeki `<inertial>` etiketlerinin eksik veya sıfır olması SAPIEN/PhysX tarafında robotun patlamasına (NaN değerler) yol açar.
- CAD programınızdan veya `trimesh` kullanarak linklerin kütle (`mass`), kütle merkezi (`origin xyz`) ve atalet değerlerini (`ixx, iyy, izz, ixy, ixz, iyz`) mutlaka doldurun.

### C. Eklem Sınırları ve Tipleri (Joint Limits)
- Dönen eklemler için `type="revolute"` (limitli) veya `type="continuous"` (limitsiz).
- `<limit lower="..." upper="..." effort="..." velocity="..."/>` alanlarını motorlarınızın fiziksel tork (`effort`) ve hız sınırlarına uygun girin.

---

## 3. URDF Doğrulama
URDF dosyanızı `assets/urdf/mahut_kol.urdf` konumuna kaydettikten sonra, kök dizindeki doğrulama betiğini çalıştırın:
```bash
python scripts/check_urdf.py
```
