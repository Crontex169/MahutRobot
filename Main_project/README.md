# MahutKol: Ana Hat (Main_project)

Özel 5-DOF kol + paralel tutucu için ManiSkill 3 simülasyonu, teleop/scripted demo kaydı (doğrudan
LeRobot v3), Modal'da VLA eğitimi (SmolVLA / ACT) ve robotumuza göre ince ayar, kapalı çevrim
değerlendirme. Python paketi: `src/mahut_kol`.

```bash
cd Main_project
pip install -r requirements.txt && pip install -e .   # Python >=3.10, numpy<2
pytest tests/
```

- Genel kılavuz, akış, klasör yapısı ve tüm komutlar: [PROJE_REHBERI.md](PROJE_REHBERI.md)
- Görev tanımları (yönler, başarı, kameralar): [docs/VLA_GOREV_TANIMLARI.md](docs/VLA_GOREV_TANIMLARI.md)
- Donanım: [docs/DONANIM_BILGILERI.md](docs/DONANIM_BILGILERI.md)
- Eğitim devri: [docs/EGITIM_DEVIR_KILAVUZU.md](docs/EGITIM_DEVIR_KILAVUZU.md)
- Geliştirme günlüğü: [YAPILANLAR.md](YAPILANLAR.md)
