# Görev: VLA eğitim verisi toplama hattı (ManiSkill, çok şekilli ilişkisel yerleştirme)

Çalışma dizini: `/home/deniz/Ortak/MahutKol`. Türkçe yaz (yorum, dokümantasyon, commit gövdesi); commit başlıkları conventional commits (`feat(env): ...`).
Önce şunları oku: `CLAUDE.md`, `YAPILANLAR.md`, `PROJE_REHBERI.md`, `src/mahut_kol/envs/multi_shape_task.py`, `src/mahut_kol/envs/shape_objects.py`, `src/mahut_kol/envs/pick_cube_task.py` (fk/ik/scripted_actions), `src/mahut_kol/speech/command_parser.py` (çıktı JSON'u), `scripts/collect_demos.py`, `scripts/eval_policy.py`, `Side_project/webxr_teleop.py`, `Side_project/ik_solver.py`, `Side_project/index.html`.

## Kesin kurallar
- **`Side_project/` içindeki hiçbir dosyayı değiştirme, taşıma, silme.** Oradaki kodu okuyup referans al; yeni kodu ana pakette (`src/mahut_kol/`) yaz.
- Yeni kodu **`datasets/` adlı bir klasöre koyma** (`.gitignore` yok sayıyor).
- Kayıt/veri dosyalarını **asla silme veya üzerine yazma**: veri kümesi her zaman var olana eklenir.
- Python ortamı: `~/.venvs/lerobot/bin/python` (lerobot 0.4.4 + smolvla kurulu, sistem paketlerini görür). Sistemde numpy 2.2.6 var, projede `numpy<2` notu var: sorun çıkarsa bildir, kendi başına ortamı bozma.
- `nvidia-smi` sürücü uyumsuzluğu veriyor (yeniden başlatma bekliyor), PyTorch CUDA yine de çalışıyor. Bu görevin parçası değil.
- Mevcut testler (`pytest tests/`) bozulmamalı; yeni özellik için test ekle.
- Eğitim, model seçimi, Modal ve doğruluk testleri **bu görevin kapsamı dışında**. Önce eğitim verisi toplama hattı çalışsın. Yollar parametrik olsun (yerel/Modal ayrımı sonra).
- Gerçek donanım yok, her şey simülasyonda. Sim-to-real sonraya kalıyor.

## Büyük resim
Sesli Türkçe komut -> `CommandParser` JSON'u -> sabit İngilizce talimat -> VLA. Eğitim metni İngilizce şablondur (ham Türkçe/STT metni eğitime girmez). Sesten gelen JSON örneği:
```
{"action":"pick_and_place","manipulated_object":"blue_cube","reference_object":"red_cylinder",
 "relation":"RIGHT","target_slot":"red_cylinder_RIGHT","valid":true,...}
```
Ortam `MultiShapeRelationalTask` (`MahutKol-MultiShapeRelational-v1`): 9 nesne (3 renk x 3 şekil), reset `options`: `manipulated_object`, `reference_object`, `relation`, `num_distractors`. Kayıtta **2 distractor** (toplam 4 aktif nesne) kullanılır.
Veri iki kaynaktan gelecek: (1) kullanıcının telefonla (WebXR) teleop kayıtları (~150-300 bölüm), (2) otomatik scripted demolar (binlerce, dengeli kapsama). Aynı şema, sidecar'da `source: teleop|scripted`.

## Aşama 1 - Ortam (multi_shape_task.py ve ilgili dosyalar)
1. **Çalışma alanını genişlet.** Mevcut sabitler SO-100'den kalma (x 0.12-0.22, y -0.10..0.10). MahutKol'un erişilebilir alanını FK/IK ile **örnekleyerek hesapla** (tahmin etme), güvenli bir iç dikdörtgeni seç ve sabitlere yaz. `ROBOT_REST_QPOS` ve SO-100 yorumlarını MahutKol'a göre düzelt.
2. **Hedef mesafesi** `DEFAULT_RELATIONAL_DELTA` 0.045 -> **0.06 m**. Komşu yönlerin başarı bölgeleri örtüşmesin (tolerans buna göre ~0.02-0.025 m; gerekçeyi yaz). `MIN_PAIRWISE_DISTANCE` ve nesne boyutları ile tutarlılığı kontrol et.
3. **TOP kuralı:** referans nesne `triangular_prism` ve ilişki `TOP` ise `ValueError` (açık Türkçe mesaj) fırlat. Komut örnekleyici bu kombinasyonu hiç üretmesin. Manipüle nesne ile referans nesne aynı olamaz. Geçerli kombinasyon sayısı 9x8x5 - 24 = **336** olmalı (doğrula).
4. **Hedef noktası boş olmalı:** hedef pozisyonu başka bir aktif nesneyle (distractor) çakışıyorsa distractor'ları yeniden örnekle (çakışmasız, `MIN_PAIRWISE_DISTANCE`'a uyarak). Hedefin çalışma alanı içinde kaldığını garanti et.
5. **Başarı kriteri:** nesne hedefe tolerans içinde **ve** bırakılmış olmalı (kıskaç açık, nesne neredeyse durağan: hız eşiği altında, birkaç ardışık adım). Sadece anlık mesafe yetmez.
6. **Süre:** `max_episode_steps` kayıt için 600'e çıkar (parametrik).
7. **Kameralar (tek yerden ayarlanan sabitler, geçici):** sabit kamera (`front`) + kıskaç üstüne monte bilek kamerası (`wrist`), ikisi **224x224**. Sabit kamerayı operatör tarafına (robota bakacak şekilde, örn. x>0'dan -x yönüne bakan) koy ki **görüntünün sağı = +y = RIGHT** olsun; FRONT = robota doğru (-x). Render alıp kontrol et, görüntüyü `docs/` altına kaydet, tanımları (FRONT/BACK/LEFT/RIGHT/TOP) dokümana yaz.
8. Geriye dönük uyumluluk: mevcut ortam kimlikleri ve `tests/test_multi_shape_task.py`, `test_shape_objects.py` vb. geçmeli (gerekirse testleri yeni sabitlere göre güncelle ve nedenini yaz).

## Aşama 2 - Komut katmanı (`src/mahut_kol/commands/` gibi bir paket, `datasets/` DEĞİL)
- Komut şeması: parser JSON'undan (`manipulated_object`, `reference_object`, `relation`) `reset(options)` üretimi.
- `to_english_instruction(cmd)`: sabit şablonlar ve birkaç paraphrase (örn. "pick the blue cube and place it to the right of the red cylinder", "put the blue cube on the right side of the red cylinder", TOP için "put ... on top of ..."). Paraphrase seçimi seed'li.
- `to_slots(cmd)`: 5 sayısal slot (manipüle renk, şekil; referans renk, şekil; yön).
- **Dengeli komut kuyruğu:** rastgele sıra yerine kapsama garantili örnekleyici: her (manipüle nesne x yön) ve (referans nesne x yön) hücresi dengeli görülsün; karıştırılmış kuyruk, devam ettirilebilir durum dosyası, seed.
- **Held-out kümesi:** 336 geçerli kombinasyondan ~30 tanesi (kapsama bozulmayacak şekilde seçilerek) sabit bir JSON'a yazılır; kayıtta atlanır (`--include-heldout` ile açılır). Değerlendirme için ayrılır.
- İsteğe bağlı komut kaynağı: `queue` (varsayılan), `text` (Türkçe cümle -> `CommandParser`), `voice` (`VoiceCommander.listen_and_parse`). Ses/metin geçersiz çıkarsa (`valid=False`) bölümü başlatma.

## Aşama 3 - Telefon teleopu (ManiSkill için yeni dosya)
Yeni dosya, örn. `src/mahut_kol/teleop/phone_webxr.py` (+ `src/mahut_kol/teleop/web/index.html` kopyası; orijinale dokunma). `Side_project/webxr_teleop.py` MuJoCo/SO-100 içindir, ona dokunmadan şunu uyarla:
- Aynı FastAPI + uvicorn WebSocket protokolü (`type: teleop` -> `clutch`, `delta_pos`, `delta_euler`, `gripper`; `type: reset`), daemon thread, thread-safe paylaşılan durum, `adb reverse tcp:8000 tcp:8000` kullanım bilgisi.
- MuJoCo döngüsü yerine ManiSkill `env.step` döngüsü. IK: MahutKol 5-DoF (joint0 yaw, joint1-3 pitch, joint4 roll) için (nullspace'li DLS çözücüyü uyarlayabilir ya da `pick_cube_task.py` fk/ik'ini genişletebilirsin; ikisini karşılaştırıp birini seç). Eksen eşlemesini ManiSkill dünya eksenlerine göre (robot +x'e bakıyor) yeniden tanımla ve dokümante et. Çalışma alanı kırpma, adım sınırlama, EMA, debriyaj (bırakılınca kol sabit kalır) korunur.
- Eylem: 7 boyutlu `pd_joint_pos` hedefi (5 eklem + 2 parmak, kıskaç kaydırıcı 0-1 -> parmak aralığı). Kontrol frekansı ortamın `control_freq`'ine (20 Hz) uyar, telefon paketleri daha sık gelir, son paket kullanılır.
- Cihazdan bağımsız arayüz (`ActionSource`): telefon, PS4/klavye (mevcut `Teleoperasyonlar/` kontrolcüleri) aynı kayıt betiğine takılabilsin. Telefon yoksa klavye/PS4 ile de kayıt yapılabilsin.
- Telefonsuz test: sahte WebSocket mesajlarıyla IK/akış birim testi, IK'nın çalışma alanındaki hedeflere ulaştığını doğrulayan test.

## Aşama 4 - Kayıt betiği (`scripts/record_vla_demos.py`)
- Her bölüm: komut seç -> `reset(options)` -> talimatı ekranda/terminalde göster -> teleop -> başarı otomatik algılanır; kullanıcı onaylar (Enter kaydet, R at ve yeniden dene, Q çık).
- **LeRobotDataset (v0.4.4 API) formatında** kaydet. Özellikler: `observation.images.front` (224x224x3 video), `observation.images.wrist` (224x224x3 video), `observation.state` (7), `action` (7), `task` = İngilizce talimat; ayrıca 5 slotluk komut vektörünü bir özellik olarak ya da meta olarak sakla. fps 20.
- Varsayılan kök: `~/vla_data/mahut_multishape`, repo_id `local/mahut_multishape` (argümanla değişir).
- **Devam/ekleme:** var olan veri kümesini açıp bölüm eklemeyi destekle (0.4.4'te çalıştığını doğrula, çalışmıyorsa güvenli bir yol bul). `shutil.rmtree` YOK. Şema uyuşmazlığında hata ver, dokunma.
- Yan dosya (sidecar JSONL, örn. `<root>/mahut_episodes.jsonl`): bölüm indeksi, seed, komut JSON'u, İngilizce talimat, held-out bayrağı, `source`, başarı, süre.

## Aşama 5 - Otomatik (scripted) demo üretici (`scripts/generate_scripted_demos.py`)
- `pick_cube_task.scripted_actions` mantığını çok şekilli göreve genişlet: nesne tutma (şekle göre yaklaşma/yükseklik, prizma için yönelme), kaldır, hedefin üstüne taşı, indir, bırak, geri çekil; distractor'lara çarpmadan.
- Aynı kayıt şeması ve komut kuyruğu; yalnızca başarılı bölümler kaydedilir, `source: scripted`. Başarı oranını ve hata nedenlerini raporla. Çok sayıda bölümde ilerleme ve devam edebilme.

## Aşama 6 - Doğrulama ve kapsama raporu
- `scripts/validate_dataset.py`: şema, kare sayıları, NaN, video okunabilirliği, sidecar tutarlılığı, (manipüle nesne x yön) ve (referans nesne x yön) kapsama tablosu, held-out sızıntı kontrolü.

## Aşama 7 - Değerlendirme betiği düzeltmesi (`scripts/eval_policy.py`)
- State artık 7 boyutlu (`qpos[:, :6]` hatası düzelt), iki kamera, `reset(options)` komut JSON'undan; talimat şablonu aynı.
- Çıktı: nesne/yön bazında başarı, held-out kombinasyonlarda başarı, aynı sahneyi farklı komutlarla çalıştırıp dili yok sayıp saymadığını ölçen test. Burada sadece betik düzeltilir, değerlendirme koşulmaz.

## Aşama 8 - Dokümantasyon
- `PROJE_REHBERI.md`: kamera planı (sabit + kıskaç üstü bilek kamerası, çözünürlükler geçici: 224x224, donanım netleşince güncellenecek), yeni betikler, yön tanımları, komut/veri akışı. `YAPILANLAR.md`: yeni bölüm. `docs/MAHUTKOL_PIPELINE.md`: silinmiş `datasets/`, `train/`, `eval/` stub referanslarını güncelle. `CLAUDE.md`'yi yalnızca gerçekten değişen komut/mimari bilgisiyle güncelle.

## Çalışma düzeni
- Aşamaları sırayla yap, her aşamada test ve çalıştırarak doğrula (ortam için render al, IK için sayısal test, kayıt için küçük bir bölüm kaydedip validate betiğiyle kontrol et). Aşama başına bir commit.
- Telefon donanımı sana erişilebilir değil: telefon kısmını sahte mesajlarla test et, gerçek telefon testini kullanıcıya bırak ve nasıl deneyeceğini adım adım yaz.
- Kararsız kaldığın nokta yalnızca kullanıcıya özgü bir tercihse sor; aksi halde makul varsayılanı seç, nedenini yaz ve ilerle.
