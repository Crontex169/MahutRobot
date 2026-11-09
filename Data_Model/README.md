# Data_Model

Eğitimi kişiler arasında devretmek için **Drive'daki klasörle aynı düzen** (içindeki tar dosyaları git'e girmez):

```text
Data_Model/
├── README.md
├── sabit/sabit_v1.tar                       # bir kez indirilir, değişmez
└── iterasyonlar/<hat>/iter_0001_step_005000_deniz.tar
```

Kısa kullanım (ayrıntı: [docs/EGITIM_DEVIR_KILAVUZU.md](../docs/EGITIM_DEVIR_KILAVUZU.md)):

```bash
python cloud/devir.py durum --hat act_v1
python cloud/devir.py kur                                      # bir kez: sabit tar'ı kendi Modal Volume'unuza
python cloud/devir.py devam  --hat act_v1 --kisi <adınız> --adim 5000
python cloud/devir.py paketle --hat act_v1 --kisi <adınız>      # çıkan .tar'ı Drive'a yükleyin
```

Herkes **kendi** Modal hesabını kullanır (`modal setup`); token paylaşılmaz. Kod Drive'a konmaz: herkes
aynı git commit'inden çalışır.
