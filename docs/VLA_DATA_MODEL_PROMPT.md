# Görev: `Data_Model` ile eğitimi kişiler arasında devretme sistemi

Çalışma dizini: `/home/deniz/Ortak/MahutKol`. Yorum ve dokümantasyon Türkçe; commit başlıkları conventional commits.
Önce oku: `CLAUDE.md`, `PROJE_REHBERI.md` §9, `YAPILANLAR.md` §16, `cloud/modal_app.py`, `cloud/train_entry.py`, `cloud/mahut_cloud.py`, `configs/training/*.yaml`, `scripts/make_act_dataset.py`.

`feat/modal-egitim-hatti` dalından `feat/data-model-devir` dalını aç. Aşama başına commit at. Push ve PR'ı yalnızca kullanıcı isterse yap.

## Amaç
Bir eğitimi proje ekibinden kişiler sırayla sürdürecek. Herkes **kendi** Modal hesabını kendi bilgisayarında kullanır. Model Google Drive'da durur. Sırası gelen kişi son modeli indirir, tek komutla eğitimi sürdürür, çıkan yeni dosyayı Drive'a geri yükler.

Dosyalar Drive'a tarayıcıdan elle yüklenip indirilir. `rclone` veya Drive API entegrasyonu ekleme.

## Kesin kurallar
- `Side_project/` içindeki hiçbir dosyaya dokunma. Yeni kodu `datasets/` adlı klasöre koyma.
- **Hesap paylaşımı yok.** Sistem başkasının Modal anahtarıyla çalışmayı kolaylaştırmaz: token saklama, profil değiştirme veya birden çok hesabı sırayla kullanma özelliği ekleme. Her kişi `modal setup` adımını kendisi yapar. Manifest, tar, log ve dokümanlarda token veya e-posta bulunmaz; yalnızca `--kisi` ile verilen kısa ad yazılır.
- Kart bilgisi kimsenin hesabına girilmeyecek; sistem yalnızca ücretsiz krediyle çalışacak şekilde tasarlanır.
- Yerel veriyi (`~/vla_data/*`) ve var olan `iter_*.tar` dosyalarını asla silme veya üzerine yazma. Kullanıcının gerçek verisi `~/vla_data/mahut_multishape` bu görevde okunmaz ve yüklenmez; testlerde `mahut_smoke_v1` kullan.
- Bu görevde toplam Modal harcaman kullanıcının onayı olmadan **1 USD**'yi geçmesin. Testler ACT ve az adımla yapılır. Uzun eğitim başlatma.
- `pytest tests/` bozulmamalı. Modal hesabı gerektirmeyen mantığın testini yaz.

## Dosya düzeni
```text
Data_Model/                     # git'e girmez; Drive'daki klasörle aynı düzen
├── README.md                   # git'te tutulur, kısa kullanım
├── sabit/
│   └── sabit_v1.tar            # bir kez indirilir, değişmez
└── iterasyonlar/
    └── <hat>/                  # ör. smolvla_v1, act_v1 (her model/deney hattı ayrı)
        ├── iter_0001_step_005000_deniz.tar
        └── iter_0002_step_010000_ali.tar
```
`.gitignore`: `Data_Model/*` yok say, `!Data_Model/README.md` hariç tut (mevcut `data/` kuralı bu klasörü yakalamıyor). `Data_Model/` altında çalışıp `git status` ile hiçbir model dosyasının görünmediğini doğrula.

**`sabit_vN.tar`:** eğitim veri kümesi (`meta`, `data`, `videos`, sidecar) ve varsa ACT türevi; videolar iki kümede ortaksa tek kopya tutulacak şekilde paketle. İçindeki `manifest.json`: sürüm, veri kümesi adları, bölüm/kare sayısı, dosya özetleri (`sha256`). Veri değişirse yeni `sabit_v2.tar` üretilir, eskisi durur.

**`iter_*.tar` (sıkıştırmasız, tek dosya):** `pretrained_model/`, `training_state/`, `train.log`, `attempts.jsonl`, `manifest.json`. Manifest alanları: format sürümü, hat, iterasyon no, adım, kişi, **devam ettiği iterasyon (`parent`)**, politika, git commit, `configs/training` özeti, `sabit` sürümü, paket sürümleri (Modal imajından), kullanılan GPU, tur süresi ve her dosyanın `sha256` değeri.

Kod Drive'a konmaz: herkes aynı git commit'inden çalışır.

## Devir komutu (`cloud/devir.py`)
Bu betik yalnızca `modal`, `pyyaml` ve standart kütüphaneyle çalışmalı. Arkadaşların bilgisayarında lerobot, ManiSkill, torch veya GPU gerekmemeli. Linux dışı sistemlerde de bozulmaması için yollarda `pathlib` kullan.

| Komut | İş |
|---|---|
| `durum --hat H` | `sabit`, yerel son iterasyon, Modal profili (kimlik göstermeden), bu ayki harcama, sıradaki iterasyon no ve yapılabilecek azami adım |
| `sabit-olustur --surum 1 --veri ... [--act-veri ...]` | `sabit_v1.tar` üretir |
| `kur [--sabit Data_Model/sabit/sabit_v1.tar]` | Tar'ı doğrular, kişinin kendi Volume'una yükler (bir kez; varsa ve özet aynıysa atlar) |
| `basla --hat H --config ... --kisi K --adim N` | Ebeveynsiz ilk tur |
| `devam --hat H --kisi K --adim N` | Son `iter_*.tar` dosyasını doğrular, Volume'a geri yükler, hedef = mevcut adım + N olarak eğitimi başlatır |
| `paketle --hat H --kisi K` | Eğitim bitince tam checkpoint'i indirir, tar'ı üretir, doğrular, Drive'a yüklenecek dosyanın yolunu yazar |

### `restore` mantığı (asıl eksik parça)
- Checkpoint'lerin `train_config.json` dosyasında mutlak yollar var (`/runs/runs/<koşu>/train`, `/tmp/ds/<veri>`). Bunları yeniden yazmak yerine **tüm hesaplarda aynı koşu adı ve veri adını kullan**: koşu adı = hat adı, yol `/runs/runs/<hat>/train/checkpoints/<adım>/`. Bu yaklaşımın gerçekten çalıştığını test et; çalışmazsa yalnızca yeniden yazılması gereken alanları değiştir ve nedenini yaz.
- Restore sonrası `launch.json` ve `attempts.jsonl` durumu `train_entry.prepare_run` ile uyumlu olsun. `checkpoints/last` bağı yokken lerobot'un sürdürmesini doğrula.
- Volume'da aynı adım zaten varsa: özet aynıysa atla, farklıysa hata ver. Üzerine yazma.
- Yalnızca `train_entry.checkpoint_is_complete` geçen checkpoint paketlenir ve geri yüklenir.

### Güvenlik denetimleri (`devam` başlamadan)
Herhangi biri başarısızsa eğitimi başlatma ve nedenini Türkçe yaz:
- Tar okunabiliyor, manifest'teki her `sha256` tutuyor (yarım indirme yakalanır).
- Tar üyeleri yalnızca beklenen yollarda ve türlerde: `.safetensors`, `.json`, `.jsonl`, `.log`. Pickle dosyası (`.pt`, `.pth`, `.bin`, `.pkl`), mutlak yol, `..` içeren yol, sembolik bağ reddedilir.
- `git` çalışma ağacı temiz (`Data_Model/` hariç); `cloud/`, `configs/training/` ve `src/mahut_kol/commands/` manifest'teki commit ile aynı.
- Modal imajındaki paket sürümleri manifest ile aynı; `sabit` sürümü ve hat eşleşiyor.
- Aynı `parent` değerine sahip veya aynı numaralı iki dosya yerelde varsa dur: çatallanma var, kullanıcı `--sec <dosya>` ile birini seçmeli. Kilit dosyası kurma; kişiler zaten birbirine haber verecek.
- Hedef `step` mevcut adımdan büyük; `--adim` kredi sınırını aşmıyor.

### Kredi koruması
- Aylık kredi 30 USD. `modal billing report` (doğru parametreleri CLI'dan doğrula) ile bu ayki gerçek harcamayı oku, güvenlik payı **5 USD** bırak.
- Adım başına maliyeti `configs/training` altında politika bazında tut (ölçülen: SmolVLA ≈ 0,20 USD/1000 adım, ACT daha ucuz). Kullanılabilir tutarı aşan `--adim` değerini reddet ve azami değeri söyle.
- Kartsız hesapta kredi bitince ne olduğu ve Volume'dan indirme yapılıp yapılamadığı Modal dokümanlarında yazmıyor. Bu yüzden tasarım krediyi bitirmemeli ve `paketle` için pay bırakmalı.
- `save_freq` varsayılanı, olası kayıp yaklaşık 15 dakikayı geçmeyecek ama Volume'u gereksiz doldurmayacak şekilde seçilsin. SmolVLA checkpoint'i 1,26 GB, ACT 592 MB. Modal Volume ücretsiz depolama sınırını dokümandan ve `modal billing report` çıktısından doğrula.
- Uzun koşular için `timeout_min` tavanı `--adim` üzerinden hesaplansın; `--max-timeout-min` kontrolü işlevsiz bırakılmasın.
- Paketlenip doğrulanmış eski checkpoint'lerin Volume'dan temizlenmesi gerekiyorsa bunu yalnızca açık bir komutla yap (`temizle --hat H --onayla`): en son checkpoint'e dokunmaz, yalnızca kişinin kendi Volume'unu etkiler, otomatik çalışmaz, yerel tar'lara ve Drive'a hiç dokunmaz.

## Test
1. Birim testler: manifest üretimi, özet doğrulama, kötü tar (yol kaçışı, pickle, yarım dosya), çatallanma, kredi hesabı, ad ve iterasyon numarası.
2. Gerçek uçtan uca deneme (`mahut_smoke_v1`, ACT, yaklaşık 100'er adım). Tek hesapla "ikinci kişi"yi taklit etmek için yalnızca teste özel bir ortam değişkeni ile ayrı Volume adları kullan (ör. `MAHUT_VOLUME_SUFFIX=_test2`); bu, gerçek çok hesaplı kullanım değildir ve sıradan kullanım için dokümante etme:
   `basla` → `paketle` → tar'ı ikinci Volume setine `kur` + `devam` → adım sayacı kaldığı yerden sürmeli, optimizer durumu yüklenmeli, loss sıçramamalı → `paketle` → `parent` doğru olmalı → çıkan checkpoint `scripts/eval_policy.py` ile yerelde hatasız açılmalı.
3. Hatalı senaryolar: yarım tar, kirli git, farklı `sabit` sürümü.
Test Volume'larını silme; adlarını raporda yaz, kullanıcı karar versin.

## Dokümantasyon
- `docs/EGITIM_DEVIR_KILAVUZU.md`: teknik olmayan bir arkadaşın adım adım izleyebileceği kılavuz: `git clone`, doğru dal/commit, küçük venv + `modal`, `modal setup`, `sabit` indirme, `kur`, son `iter` indirme, `devam`, `paketle`, Drive'a yükleme, Drive'daki ad çakışırsa ne yapılacağı, sık hatalar.
- `Data_Model/README.md` kısa kullanım. `PROJE_REHBERI.md`, `YAPILANLAR.md` yeni bölüm. `CLAUDE.md` yalnızca değişen komut ve mimari bilgisi.

## Kapsam dışı
Toplu veri üretimi, tam eğitim, model seçimi, başarı oranı optimizasyonu, Drive API, telefon teleopu.

## Bitiş raporu
Ne çalıştı, ne çalışmadı, gerçek harcama (`modal billing report`), ikinci Volume denemesinin adım/loss sürekliliği, arkadaşların çalıştıracağı tam komutlar, kullanıcının karar vermesi gerekenler (dalın `main`'e birleşmesi dahil; arkadaşlar kodu git'ten alacak). Doğrulayamadığın her şeyi, özellikle gerçek ikinci hesapla yapılamayan denemeyi, açıkça yaz.
