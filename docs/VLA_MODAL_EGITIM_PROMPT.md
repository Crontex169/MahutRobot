# Görev: Modal üzerinde VLA eğitim hattı ve uçtan uca kısa deneme

Çalışma dizini: `/home/deniz/Ortak/MahutKol`. Yorum ve dokümantasyon Türkçe; commit başlıkları conventional commits.
Önce oku: `CLAUDE.md`, `PROJE_REHBERI.md` §8, `docs/VLA_GOREV_TANIMLARI.md`, `YAPILANLAR.md` §15, `scripts/eval_policy.py`, `scripts/generate_scripted_demos.py`, `scripts/validate_dataset.py`, `src/mahut_kol/recording/`, `src/mahut_kol/commands/`, `scripts/train_smolvla.sh` (eski, yalnızca referans).

Bu görevin amacı iyi bir model eğitmek değil. Amaç, verinin Modal'da eğitilebildiğini ve çıkan checkpoint'in yerelde kapalı çevrim çalıştığını kanıtlamak. Kullanıcı saatlerce telefonla kayıt yapmadan önce bu kanıt gerekiyor.

## Kesin kurallar
- `Side_project/` içindeki hiçbir dosyaya dokunma.
- Yeni kodu `datasets/` adlı klasöre koyma (`.gitignore` yok sayıyor).
- Veri kümelerini ve eğitim çıktılarını asla silme veya üzerine yazma. `rm -rf`, `shutil.rmtree` ve Volume üzerinde silme yok. Her eğitim benzersiz bir koşu adıyla yazılır. Var olan bir hedef varsa hata ver.
- `~/vla_data/mahut_multishape` kullanıcının gerçek teleop verisidir (şu an 2 bölüm). Bu görevde ona yazma ve onu Modal'a yükleme.
- Yerel Python: `~/.venvs/lerobot/bin/python` (lerobot 0.4.4). Yerelde video okurken `video_backend="pyav"` kullan.
- Modal kimlik doğrulamasını kullanıcı yapar. API anahtarı isteme, girme, dosyaya veya loglara yazma. `~/.venvs/lerobot/bin/modal profile current` başarısızsa dur ve kullanıcıdan `modal setup` çalıştırmasını iste.
- `main` üzerinde çalışma: `feat/modal-egitim-hatti` dalını aç, aşama başına commit at. Push ve PR'ı yalnızca kullanıcı isterse yap.
- Mevcut testler (`pytest tests/`) bozulmamalı. Modal hesabı gerektirmeyen yeni mantık için test ekle.

## Maliyet ve durma noktaları
- Bu görevde yalnızca kısa denemeler çalışır. Her Modal fonksiyonunda sıkı `timeout` olsun (kısa deneme için en fazla 30 dakika).
- GPU tipi parametre olsun, kısa deneme için ucuz ve yeterli bir tip seç. GPU adlarını, fiyatları ve API'yi hafızadan yazma, güncel Modal dokümantasyonundan doğrula.
- Toplam harcaman kullanıcının onayı olmadan yaklaşık 5 USD'yi geçmesin. Geçecek gibiyse dur ve sor.
- **Aşama 5 bitince dur ve raporla.** Uzun eğitim, toplu veri üretimi ve model seçimi kullanıcının onayı olmadan başlamaz.

## Aşama 1 - Deneme veri kümesi
`generate_scripted_demos.py` ile **ayrı bir köke** (`~/vla_data/mahut_smoke_v1`, repo_id `local/mahut_smoke_v1`) 200-300 scripted bölüm üret (yaklaşık 30-40 dakika, arka planda çalıştır). `validate_dataset.py` ile doğrula. Teleop ve scripted bölüm uzunluklarını karşılaştıran kısa bir özet çıkar (teleop ~21 sn, scripted ~8 sn görünüyor); bu fark sonraki kararlar için raporda yer alsın, davranışı bu görevde değiştirme.

## Aşama 2 - Modal altyapısı (`cloud/` gibi yeni bir klasör)
- Modal image: yerel ile aynı Python ve sürümler (lerobot 0.4.4 ve smolvla bağımlılıkları sabit sürümlü), ffmpeg. Image'in yerel venv ile farklarını not et.
- Volume'lar: veri için `mahut-datasets`, çıktılar ve model önbelleği için `mahut-runs`. HF model ağırlıkları Volume'da önbelleklensin.
- Betikler: veri yükleme (sürümlü ad, var olanın üzerine yazmaz), eğitim başlatma (`--detach` ile kapanan terminalden etkilenmez), checkpoint indirme (yerelde `~/vla_runs/<koşu_adı>/`).
- Eğitim checkpoint'leri periyodik olarak Volume'a kalıcı yazılsın. Kesilen bir koşu aynı koşu adıyla kaldığı yerden devam edebilsin (preemption ve timeout için).
- Koşu ayarları tek yerde olsun (`configs/training/<politika>.yaml` gibi): politika, adım, batch, kaydetme sıklığı, GPU. Yollar parametrik olsun, wandb varsayılan kapalı.

## Aşama 3 - Politika tanımları
İlk kıyas iki politikadır:

**SmolVLA (dil koşullu):** `lerobot/smolvla_base` ile ince ayar. `front` ve `wrist` kameralarını modelin beklediği adlara eşle, eksik kamerayı `empty_cameras` ile doldur. Adları ve beklenen boyutları 0.4.4 kodundan doğrula. State ve action 7 boyutlu, action mutlak `pd_joint_pos`.

**ACT (yapılandırılmış komut, dilsiz kıyas):** ACT dil almaz. 5 slotu 17 boyutlu one-hot vektöre çevir (renk 3, şekil 3, renk 3, şekil 3, yön 5) ve `observation.environment_state` olarak ver. ACT'in bu girdiyi 0.4.4'te desteklediğini doğrula. Kaynak veriyi değiştirme; ince bir eğitim sarmalayıcısı ya da ayrı sürümlü türetilmiş veri kümesi kullan, en az lerobot yaması gerektireni seç ve nedenini yaz.

Doğrulanacak riskler:
- `command.slots` (int64) özel alanı dataset stats, collate veya politika tarafında hata çıkarıyor mu? Çıkarıyorsa kaynak veriyi değil sarmalayıcıyı değiştir.
- Modal'da video okuma arka ucu açıkça seçilsin (torchcodec veya pyav).

## Aşama 4 - Kısa eğitim denemeleri
Her iki politika için:
1. **Duman testi:** 200-500 adım. Başarı ölçütü: hatasız bitiyor, loss makul, checkpoint Volume'a yazılıyor.
2. **Devam testi:** Aynı koşu adıyla checkpoint'ten sürdür, adım sayacı doğru devam etsin.
3. **Ezber testi:** ~10 bölüm üzerinde birkaç bin adım. Loss belirgin düşmeli; veri, normalizasyon veya eylem eşlemesinde hata varsa burada görünür.

Her koşu için süre, kullanılan GPU, tahmini maliyet ve adım/sn kaydet.

## Aşama 5 - Yerel kapalı çevrim doğrulama
Checkpoint'leri indir ve `scripts/eval_policy.py --checkpoint <yol>` ile yerelde ~5 bölüm koştur. Başarı oranı beklenmiyor; ölçüt, betiğin kamera adı, boyut, normalizasyon ve komut girdisi hatası vermeden sonuna kadar çalışması ve eylemlerin makul aralıkta olması.
- `eval_policy.py`'yi gerektiği kadar düzelt: SmolVLA kamera eşlemesi ve `empty_cameras`, ACT için 17 boyutlu one-hot girdi. Eğitimdeki ön işleme ile birebir aynı olsun.
- RTX 4060 (8 GB) üzerinde çıkarım için en yüksek GPU belleğini ve döngü hızını ölç.
- Daha büyük modeller (pi0 ailesi) için yerel belleğin yetip yetmeyeceğini ve Modal'da SAPIEN render'ının mümkün olup olmadığını kısa bir fizibilite notu olarak yaz. Yerel yol engellenmedikçe Modal'da değerlendirme kurma.

## Aşama 6 - Temizlik ve dokümantasyon
- Yeni hat çalıştıktan sonra `scripts/train_smolvla.sh` dosyasını kaldır (`rm -rf "$OUT"` içeriyor ve eski PickCube verisine bağlı). `scripts/collect_demos.py` kalsın.
- `PROJE_REHBERI.md`: yeni bölüm (Modal kurulumu, veri yükleme, eğitim, devam, checkpoint indirme, değerlendirme). `YAPILANLAR.md`: yeni bölüm, ölçülen sürelerle. `CLAUDE.md`: yalnızca değişen komut ve mimari bilgisi.

## Kapsam dışı
Toplu veri üretimi, tam eğitim, model seçimi, başarı oranı optimizasyonu, telefon teleopu ve gerçek robot.

## Bitiş raporu
Kullanıcıya şunları yaz: ne çalıştı ve ne çalışmadı, iki politikanın duman, devam ve ezber sonuçları, yerel değerlendirme sonucu, toplam harcama, kullanılacak komutlar, teleop/scripted uzunluk farkı için öneri, 2000 bölümlük tam eğitim için süre ve maliyet tahmini ve senden beklenen karar. Çalıştırmadığın veya doğrulayamadığın her şeyi açıkça belirt.
