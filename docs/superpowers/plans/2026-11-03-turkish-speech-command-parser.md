# Türkçe Sesli Komut (STT) ve Robot Komut Çözücü Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** MahutKol robot projesi için yerel (offline), ücretsiz ve hafif çalışan Türkçe Speech-to-Text (`faster-whisper`), ses yakalama (`sounddevice`) ve ManiSkill simülasyonuna uyumlu kural tabanlı Türkçe komut çözümleyici (`CommandParser`) modülünün kurulması.

**Architecture:** Modül 4 ana katmandan oluşur: `AudioRecorder` (16 kHz mono mikrofondan ses yakalama, Push-to-Talk ve VAD destekli), `STTEngine` (faster-whisper CTranslate2 sarmalayıcısı, GPU CUDA float16 ve CPU int8 destekli), `CommandParser` (Türkçe morfolojik ekleri, renkleri, şekilleri, eylemleri ve 5 uzamsal ilişkiyi JSON slotlarına dönüştüren deterministik çözümleyici) ve `VoiceCommander` (bu 3 bileşeni tek bir orkestrasyon sınıfında birleştiren ana arayüz).

**Tech Stack:** Python 3.10+, `faster-whisper` (CTranslate2), `sounddevice` + `numpy`, `torch` (CUDA 12.4), `json`, `re`.

**Spec:** User request: "GÖREV: Yerel (Offline) Türkçe Speech-to-Text ve Robot Komut Çözücü Modülünün Geliştirilmesi".

## Global Constraints

- STT Motoru `faster-whisper` kullanılacak, dil `language="tr"` olacak, model varsayılan `base`, isteğe bağlı `small` ve `tiny` seçilebilecek.
- Donanım uyumluluğu: CUDA varsa `float16` veya `int8_float16`, yoksa CPU üzerinde `int8` hesaplama tipiyle çalışmalı.
- Ses kaydı standardı: 16 kHz, Mono, 16-bit PCM standardı (veya float32 normalize [-1, 1]).
- Giriş modu: Hem Push-to-Talk (Bas-Konuş / Tuşa basınca dinle) hem de Sessizlik Tespiti (VAD / Energy-based) desteklenecek.
- Nesne ve yön eşleştirmeleri ManiSkill `MultiShapeRelationalTask` standartlarına tam uyumlu olacak: Renkler (`red`, `green`, `blue`), Şekiller (`cube`, `cylinder`, `triangular_prism`), Bağıl Yönler (`FRONT`, `BACK`, `LEFT`, `RIGHT`, `TOP`), Eylemler (`pick_and_place`, `pick`, `push`, `place`).

## Review Focus

1. **PortAudio kütüphanesi eksikliği:** Sistem seviyesinde libportaudio kurulu olmadığında `sounddevice` çökebilir -> `audio_recorder.py` ve `__init__.py` içinde kullanıcı dizinindeki (`~/.local/lib/libportaudio.so.2`) kütüphaneyi otomatik keşfeden ve yükleyen sağlam fallback mekanizması sağlanmalı.
2. **Türkçe büyük/küçük harf dönüşümü (`İ`/`i`, `I`/`ı`):** Standart Python `.lower()` fonksiyonu 'İ' harfini 'i\u0307' veya beklenmedik formlara çevirebilir -> `turkish_lower()` fonksiyonu ile Türkçe karakterler doğru şekilde normalize edilmeli.
3. **Türkçe çekim ekleri (Akkusatif, Yönelme, Tamlayan):** "küpü", "küpe", "küpten", "silindirin", "prizmanın", "sağına", "önüne" gibi eklerin gövdeye doğru ayrıştırılması -> regex ve kök eşleme kuralları ile hatasız eşleştirme.
4. **CUDA OOM veya cuDNN hatası:** GPU belleği yetersizse veya CUDA kütüphanesi uyumsuzsa STT motoru çökmek yerine zarifçe CPU `int8` moduna fallback yapmalı.
5. **Eksik veya geçersiz komutlar:** Kullanıcı sadece "Kırmızı küpü" veya "Sağa git" dediğinde veya anlaşılmayan bir girdi olduğunda parser çökmemeli; `valid: false` ve açıklayıcı hata mesajı dönmeli.

---

### Task 1: Türkçe Karakter ve Komut Çözücü Modülü (`CommandParser`) ve Testleri

**Files:**
- Create: `src/mahut_kol/speech/command_parser.py`
- Test: `tests/test_command_parser.py`

**Interfaces:**
- Consumes: None (saf Python standart kütüphaneleri `re`, `json`, `typing`).
- Produces: `CommandParser` sınıfı, `CommandParser.parse(text: str) -> Dict[str, Any]`, `turkish_lower(text: str) -> str`.

- [ ] **Step 1: Write the failing test for `CommandParser`**

Yazılacak testler şunları doğrular:
- "Kırmızı küpü al yeşil silindirin sağına koy" -> `action: pick_and_place`, `manipulated: red_cube`, `reference: green_cylinder`, `relation: RIGHT`, `valid: true`.
- "Mavi silindiri it" -> `action: push`, `manipulated: blue_cylinder`, `reference: None`, `relation: None`, `valid: true`.
- "Yeşil prizmayı kaldır" -> `action: pick`, `manipulated: green_triangular_prism`, `valid: true`.
- Tüm renkler (`kırmızı`, `yeşil`, `mavi`) ve şekiller (`küp`, `silindir`, `üçgen prizma` / `prizma`) kombinasyonları (9 nesne).
- Tüm bağıl yönler: `sağına` (RIGHT), `soluna` (LEFT), `önüne` (FRONT), `arkasına` (BACK), `üstüne` / `üzerine` (TOP).
- Türkçe büyük harf "KIRMIZI KÜPÜ AL" ve noktalama işaretleri temizliği.
- Eksik nesne veya yön durumunda `valid: false`.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_command_parser.py`
Expected: FAIL (ModuleNotFoundError / command_parser not found).

- [ ] **Step 3: Implement `CommandParser` and `turkish_lower` in `src/mahut_kol/speech/command_parser.py`**

- Türkçe küçük harf dönüşümü (`İ` -> `i`, `I` -> `ı`).
- Renk haritası: `kırmızı`/`al` -> `red`, `yeşil` -> `green`, `mavi` -> `blue`.
- Şekil haritası: `küp*` -> `cube`, `silindir*` -> `cylinder`, `üçgen prizma*`/`prizma*`/`üçgen*` -> `triangular_prism`.
- Yön haritası: `sağ*` -> `RIGHT`, `sol*` -> `LEFT`, `ön*` -> `FRONT`, `arka*` -> `BACK`, `üst*`/`üzer*`/`tepe*` -> `TOP`.
- Eylem haritası: `al ... koy*` / `koy*` / `yerleştir*` / `bırak*` / `taşı*` -> `pick_and_place`, `al*` / `kaldır*` / `tut*` -> `pick`, `it*` / `itele*` / `kaydır*` -> `push`.
- Çıktı: Standart ManiSkill / MahutKol nesne kimlikleri (`red_cube`, vb.) ve doğrulanmış JSON formatı.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_command_parser.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mahut_kol/speech/command_parser.py tests/test_command_parser.py
git commit -m "feat(speech): implement Turkish command parser and normalization with tests"
```

---

### Task 2: Ses Kayıt ve Dinleme Modülü (`AudioRecorder`)

**Files:**
- Create: `src/mahut_kol/speech/audio_recorder.py`
- Test: `tests/test_audio_recorder.py`

**Interfaces:**
- Consumes: `sounddevice`, `numpy`, `ctypes.util` (PortAudio fallback).
- Produces: `AudioRecorder` sınıfı.
  - `record_push_to_talk(prompt_msg: str = ...) -> np.ndarray`
  - `record_fixed_duration(duration_seconds: float) -> np.ndarray`
  - `record_vad(silence_threshold: float, silence_duration: float, max_duration: float) -> np.ndarray`
  - `save_wav(audio_data: np.ndarray, file_path: str)`

- [ ] **Step 1: Write the test for `AudioRecorder`**

Yazılacak testler:
- PortAudio kütüphanesi otomatik keşif kontrolü.
- Enerji/RMS eşiği hesaplama fonksiyonu testi (`compute_rms`).
- VAD durum makinesi mantığının (simüle ses verisiyle) testi: sessizlik -> konuşma -> sessizlik döngüsü sonlandırma.
- WAV dosyası kaydetme ve geri okuma format doğrulaması (16 kHz, Mono, float32/int16).

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_audio_recorder.py`
Expected: FAIL.

- [ ] **Step 3: Implement `AudioRecorder` in `src/mahut_kol/speech/audio_recorder.py`**

- `~/.local/lib/libportaudio.so.2` ve sistem yollarını tarayan PortAudio preload kancası.
- 16000 Hz, 1 kanal (Mono) ses yakalama standardı.
- `record_push_to_talk`: Enter'a basınca başlar, Enter'a basınca durur; yakalanan sesi numpy float32 dizisi olarak döner.
- `record_vad`: Enerji tabanlı ses aktivitesi tespiti; ses başladığında kayda başlar, belirtilen süre sessizlik olunca durur.
- Standart kütüphane `wave` modülü ile WAV kaydetme fonksiyonu.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_audio_recorder.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mahut_kol/speech/audio_recorder.py tests/test_audio_recorder.py
git commit -m "feat(speech): implement audio recorder with push-to-talk and VAD support"
```

---

### Task 3: Faster-Whisper STT Motoru Sarmalayıcısı (`STTEngine`)

**Files:**
- Create: `src/mahut_kol/speech/stt_engine.py`
- Test: `tests/test_stt_engine.py`

**Interfaces:**
- Consumes: `faster_whisper.WhisperModel`, `torch`, `numpy`.
- Produces: `STTEngine` sınıfı.
  - `__init__(model_size="base", device=None, compute_type=None, language="tr")`
  - `transcribe(audio: Union[str, np.ndarray, bytes], language="tr") -> Dict[str, Any]`

- [ ] **Step 1: Write the test for `STTEngine`**

Yazılacak testler:
- STTEngine başlatma parametreleri (cihaz seçimi, compute_type belirleme: CUDA varsa float16, yoksa CPU int8).
- CPU int8 fallback mantığı.
- Sentetik ses dizisi veya örnek dosya ile `transcribe` çağrısı ve dönen sözlük yapısı doğrulaması (`text`, `language`, `duration`, vb.).

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_stt_engine.py`
Expected: FAIL.

- [ ] **Step 3: Implement `STTEngine` in `src/mahut_kol/speech/stt_engine.py`**

- `faster_whisper.WhisperModel` sarmalama.
- Otomatik CUDA algılama: `torch.cuda.is_available()` ile RTX 4060 GPU üzerinde `float16` kullanımı.
- CUDA başlatma hatası durumunda otomatik `CPU / int8` fallback.
- Dil varsayılanı Türkçe (`language="tr"`), `beam_size=5`, `vad_filter=True`.
- NumPy ses dizilerini ve dosya yollarını doğrudan kabul etme.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_stt_engine.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mahut_kol/speech/stt_engine.py tests/test_stt_engine.py
git commit -m "feat(speech): implement faster-whisper STT wrapper with CUDA and CPU fallback"
```

---

### Task 4: Orkestrasyon Sınıfı (`VoiceCommander`) ve Paket Başlığı (`__init__.py`)

**Files:**
- Create: `src/mahut_kol/speech/__init__.py`
- Create: `src/mahut_kol/speech/voice_commander.py`
- Modify: `requirements.txt`
- Test: `tests/test_voice_commander.py`

**Interfaces:**
- Consumes: `AudioRecorder`, `STTEngine`, `CommandParser`.
- Produces: `VoiceCommander` sınıfı.
  - `listen_and_parse(mode="push_to_talk") -> Tuple[str, Dict[str, Any]]`
  - `process_text(text: str) -> Dict[str, Any]`
  - `process_audio_file(file_path: str) -> Tuple[str, Dict[str, Any]]`
  - `run_interactive_loop(callback=None, mode="push_to_talk")`

- [ ] **Step 1: Write integration tests for `VoiceCommander`**

Yazılacak testler:
- `VoiceCommander` bileşen entegrasyonu (mock STT ve AudioRecorder ile uçtan uca boru hattı).
- `process_text` doğrudan metin çözümleme testi.
- `__init__.py` üzerinden sınıfların dışa aktarımı (`from mahut_kol.speech import VoiceCommander, CommandParser, STTEngine, AudioRecorder`).

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_voice_commander.py`
Expected: FAIL.

- [ ] **Step 3: Implement `voice_commander.py` and `__init__.py`**

- `VoiceCommander` sınıfında `AudioRecorder`, `STTEngine` ve `CommandParser` örneklerini yönetme.
- Canlı dinleme ve robot komutu üretme boru hattı (`listen_and_parse`).
- `requirements.txt` dosyasına `faster-whisper` ve `sounddevice` ekleme.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_voice_commander.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mahut_kol/speech/ voice_commander.py tests/test_voice_commander.py requirements.txt
git commit -m "feat(speech): implement VoiceCommander pipeline and package exports"
```

---

### Task 5: Canlı Test ve Doğrulama Betiği (`scripts/test_voice_command.py`)

**Files:**
- Create: `scripts/test_voice_command.py`

**Interfaces:**
- Consumes: `mahut_kol.speech` (`VoiceCommander`, `CommandParser`, `STTEngine`, `AudioRecorder`).
- Produces: CLI test arayüzü (`--mode push_to_talk | vad | text | file`, `--model base | small | tiny`, `--device cuda | cpu`).

- [ ] **Step 1: Implement `scripts/test_voice_command.py`**

- Argument parser: `--mode`, `--model`, `--device`, `--audio-file`, `--input-text`, `--interactive`.
- Renkli konsol çıktıları (Yeşil: Başarılı komut, Mavi: Transkript, Kırmızı: Hata).
- Komut ayrıştırma detaylarını (nesne, eylem, yön, güvenilirlik) tablo/JSON formatında temiz şekilde görüntüleme.

- [ ] **Step 2: Verify `scripts/test_voice_command.py` with mock/text mode and audio mode**

Run: `python3 scripts/test_voice_command.py --mode text --input-text "Kırmızı küpü al yeşil silindirin sağına koy"`
Expected: Transkripsiyon ve ayrıştırılmış geçerli JSON çıktısı.

- [ ] **Step 3: Commit**

```bash
git add scripts/test_voice_command.py
git commit -m "feat(speech): add interactive voice command CLI test tool"
```

---

### Task 6: Tam Regresyon ve Dokümantasyon Güncellemesi

**Files:**
- Modify: `README.md`
- Modify: `YAPILANLAR.md`
- Test: Bütün testler (`tests/test_*.py`)

- [ ] **Step 1: Run all test suites**

Run: `python3 -m unittest discover -s tests -p "test_*.py"`
Expected: All tests pass.

- [ ] **Step 2: Update README.md and YAPILANLAR.md**

- `README.md` içine Sesli Komut Modülü (Speech-to-Text & Parser) kullanım talimatlarını ekleme.
- `YAPILANLAR.md` içine geliştirme detaylarını kaydetme.

- [ ] **Step 3: Commit**

```bash
git add README.md YAPILANLAR.md
git commit -m "docs: document speech-to-text and voice command module"
```
