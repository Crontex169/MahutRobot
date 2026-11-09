# Eğitim Devir Kılavuzu (arkadaşlar için)

Bir robot politikasını (ACT ya da SmolVLA) sırayla eğitiyoruz. Her tur kısa sürer; sıra sana
geldiğinde **son modeli indirir, eğitimi sürdürür, çıkan yeni dosyayı Drive'a geri yüklersin.**

**Kurallar (lütfen oku):**
- Herkes **kendi** Modal hesabını kullanır. Kimseyle token, şifre ya da anahtar paylaşma, kimseninkini isteme.
- Hesabına **kart bilgisi girme**. Sistem yalnızca Modal'ın aylık ücretsiz kredisiyle çalışacak şekilde tasarlandı
  (aşağıdaki "Kredi" bölümüne bak).
- Aynı anda iki kişi aynı iterasyondan devam etmesin. Sırayı grupta haber verin (kilit dosyası yok).
- Kod Drive'da değil, git'te durur. Herkes **aynı git commit'inden** çalışır.

## 0. Bir kez yapılacak kurulum

### 0.1 Python ve küçük bir sanal ortam
Python 3.10 ya da üstü gerekir (lerobot, torch, GPU **gerekmez**).

Linux / macOS:
```bash
python3 -m venv ~/mahut-devir
source ~/mahut-devir/bin/activate
pip install "modal==1.6.3" pyyaml
```
Windows (PowerShell):
```powershell
py -m venv $HOME\mahut-devir
$HOME\mahut-devir\Scripts\Activate.ps1
pip install "modal==1.6.3" pyyaml
```
Her yeni terminal açtığında önce `source ~/mahut-devir/bin/activate` (Windows: `Activate.ps1`) komutunu çalıştır.

### 0.2 Kodu al (doğru dal / commit)
```bash
git clone https://github.com/Crontex169/MahutRobot.git
cd MahutRobot
git checkout <DAL_VEYA_COMMIT>      # bunu gruptaki sorumlu kişi söyler
git log -1 --oneline                 # söylenen commit'le aynı mı bak
```
Dosyalarda **hiçbir değişiklik yapma** (git temiz olmalı; aksi halde komutlar durur).

### 0.3 Modal hesabın
1. https://modal.com adresinden **kendi** hesabını aç.
2. Terminalde: `modal setup` → tarayıcıda giriş yap. (Anahtar bu bilgisayarda Modal'ın kendi dosyasına yazılır; sen kimseye göndermezsin.)

### 0.4 Sabit paketi indir ve kur
1. Drive'daki `Data_Model/sabit/sabit_v1.tar` dosyasını **tarayıcıdan indirip** projedeki `Data_Model/sabit/` klasörüne koy
   (klasör yoksa oluştur). Dosya adı tam `sabit_v1.tar` olmalı.
2. Kendi Modal depona yükle (bir kez; sonra "zaten kurulu" der ve atlar):
```bash
python cloud/devir.py kur
```

## 1. Her tur: sırası gelen kişi

### 1.1 Durumu kontrol et
```bash
python cloud/devir.py durum --hat act_v1
```
Gösterir: sabit var mı, yerel son iterasyon, bu ayki harcama, **yapabileceğin azami adım**.

### 1.2 Son modeli indir
Drive'daki `Data_Model/iterasyonlar/<hat>/` klasöründen **en büyük numaralı** `iter_NNNN_step_..._kisi.tar` dosyasını
indirip projedeki aynı klasöre (`Data_Model/iterasyonlar/<hat>/`) koy. (İndirmeyi bitirmeden başlatma; yarım dosya
otomatik yakalanır ve komut durur.)

### 1.3 Eğitimi sürdür
```bash
python cloud/devir.py devam --hat act_v1 --kisi <kısa_adın> --adim 5000
```
- `--kisi`: küçük harf kısa adın (ör. `ali`). E-posta ya da boşluk yazma.
- `--adim`: bu turda kaç adım eğitileceği. Çok büyük verirsen sistem kredi sınırını söyleyip reddeder.
- Hesabında **ilk kez** çalıştırıyorsan Modal eğitim imajını derler (birkaç dakika sürebilir, sonra önbellekten gelir).
- Komut kontrolleri yapar, modeli kendi depona yükler ve eğitimi başlatır. Terminali kapatsan bile eğitim Modal'da sürer.
  İlerlemeyi terminalde görürsün.

### 1.4 Paketle
Eğitim "✅ Eğitim bitti" dediğinde:
```bash
python cloud/devir.py paketle --hat act_v1 --kisi <kısa_adın>
```
Çıktıda yeni dosyanın yolu yazar (ör. `Data_Model/iterasyonlar/act_v1/iter_0003_step_015000_ali.tar`).

### 1.5 Drive'a yükle
Çıkan `.tar` dosyasını tarayıcıdan Drive'daki `Data_Model/iterasyonlar/<hat>/` klasörüne yükle ve grupta haber ver.

**Drive'da aynı adlı dosya varsa ne yapmalı?** Dosya adında sıra numarası, adım ve kişi adı olduğu için aynı ada
genelde rastlanmaz. Drive "(1)" gibi bir ek koyarsa ya da iki kişi aynı numarayı üretirse (aynı iterasyondan iki kişi
devam etmiş = **çatallanma**):
- Yüklemeyi **ezme** ("Değiştir" deme); "Her ikisini de tut" seç.
- Grupta hangisinin devam edileceğine karar verin. Seçilen dosyadan devam etmek için:
  `python cloud/devir.py devam --hat act_v1 --kisi <ad> --adim 5000 --sec iter_0002_step_010000_veli.tar`
- Dosya adının sonuna Drive'ın eklediği ` (1)` gibi ekleri **sil** (ad tam desene uymalı).

## 2. İlk tur (yalnızca ilk kişi)
```bash
python cloud/devir.py basla --hat act_v1 --config configs/training/act.yaml --kisi <ad> --adim 5000
python cloud/devir.py paketle --hat act_v1 --kisi <ad>
```

## 3. Kredi
- Modal Starter planında ayda 30 USD ücretsiz kredi vardır. Sistem **5 USD güvenlik payı** bırakır ve bu ayki gerçek
  harcamana göre izin verilen azami `--adim`'i hesaplar.
- Kredi bittiğinde ne olacağı (çalışan işler, Volume'dan indirme) Modal belgelerinde yazmıyor. Bu yüzden krediyi bitirme;
  `paketle` için pay kalsın.
- Modal belgeleri hesap açarken ödeme yöntemi istenebileceğini söylüyor; bu konuda gruptaki sorumluyla konuş.
  Kendi kartını girmeden önce mutlaka sor.
- Depoyu temizlemek istersen (yalnızca **kendi** Modal depon): `python cloud/devir.py temizle --hat act_v1` (önce neyi
  sileceğini gösterir) ve onaylıyorsan `--onayla`. En son checkpoint'e ve Drive/yerel tar dosyalarına dokunmaz.

## 4. Sık hatalar
| Mesaj | Anlamı / ne yapmalı |
|---|---|
| `Modal profili yok` | `modal setup` çalıştır (kendi hesabınla). |
| `tar okunamıyor` / `özet (sha256) tutmuyor` | İndirme yarım ya da bozuk. Dosyayı Drive'dan yeniden indir. |
| `git çalışma ağacı temiz değil` | Dosya değiştirmişsin. `git status` ile bak, `git checkout -- <dosya>` ile geri al. |
| `... manifest'teki commit ile aynı değil` | Yanlış commit'tesin. `git checkout <söylenen commit>`. |
| `sabit_vN.tar ... içinde yok` | Drive'dan `Data_Model/sabit/` klasörüne indir. |
| `Veri kümesi ... Volume'unuzda yok` | `python cloud/devir.py kur` çalıştır. |
| `paket sürümleri ... aynı değil` | Kod commit'i farklı olabilir; doğru commit'e geç, hâlâ sürüyorsa gruptaki sorumluya bildir. |
| `ÇATALLANMA var` | Yukarıdaki "aynı ad / çatallanma" bölümü; `--sec` ile seç. |
| `--adim ... kredi sınırını aşıyor` | Mesajdaki azami adımdan küçük bir değer ver. |
| `Eğitim bitmemiş ya da başarısız` (paketle) | Önce eğitimin bitmesini bekle ya da `devam`ı yeniden çalıştır (kaldığı yerden sürer). |
| `L4/A10 kapasitesi yok` (uzun bekleme) | Modal'da GPU bulunamıyor; bir süre sonra tekrar dene. Beklerken ücret işlemez. |

Sorun çözülmezse: terminaldeki hata metnini kopyalayıp gruptaki sorumluya gönder (token ya da e-posta içermez).
