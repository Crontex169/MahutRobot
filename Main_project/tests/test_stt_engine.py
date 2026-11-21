"""
Birim testleri: Faster-Whisper Yerel STT Motoru (STTEngine).
"""

import unittest
import numpy as np
import torch

from mahut_kol.speech.stt_engine import STTEngine, resolve_device_and_compute_type


class TestSTTEngine(unittest.TestCase):
    def test_resolve_device_and_compute_type(self):
        """Donanım ve hesaplama tipi belirleme mantığını doğrula."""
        # CPU açıkça istendiğinde compute_type varsayılan int8 olmalı
        dev, comp = resolve_device_and_compute_type(device="cpu", compute_type=None)
        self.assertEqual(dev, "cpu")
        self.assertEqual(comp, "int8")

        # Özel compute_type verildiğinde korunmalı
        dev, comp = resolve_device_and_compute_type(device="cpu", compute_type="float32")
        self.assertEqual(dev, "cpu")
        self.assertEqual(comp, "float32")

        # Otomatik tespitte CUDA varsa cuda/float16 seçilmeli
        if torch.cuda.is_available():
            dev, comp = resolve_device_and_compute_type(device=None, compute_type=None)
            self.assertEqual(dev, "cuda")
            self.assertIn(comp, ("float16", "int8_float16"))

    def test_stt_engine_init_and_transcribe_dummy(self):
        """Hafif model (tiny) ile STTEngine başlatma ve boş/sentetik seste transkripsiyon yapısını doğrula."""
        # Hızlı test için tiny modeli kullanalım
        engine = STTEngine(model_size="tiny", language="tr")
        self.assertIsNotNone(engine.model)
        self.assertEqual(engine.language, "tr")

        # 1 saniyelik 16 kHz sessizlik dizisi
        audio_dummy = np.zeros(16000, dtype=np.float32)
        res = engine.transcribe(audio_dummy)

        self.assertIsInstance(res, dict)
        self.assertIn("text", res)
        self.assertIn("language", res)
        self.assertIn("device", res)
        self.assertIn("compute_type", res)
        self.assertEqual(res["language"], "tr")

    def test_empty_audio_handling(self):
        """Boş ses dizisi verildiğinde hata fırlatmak yerine boş metin dönmeli."""
        engine = STTEngine(model_size="tiny", language="tr")
        res = engine.transcribe(np.zeros(0, dtype=np.float32))
        self.assertEqual(res["text"], "")

    def test_domain_prompt_and_hotwords_kwargs(self):
        """Domain sözlüğü initial_prompt/hotwords olarak iletilmeli ve kapatılabilmeli."""
        from mahut_kol.speech.stt_engine import DOMAIN_HOTWORDS, DOMAIN_INITIAL_PROMPT

        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def transcribe(self, audio, language=None, beam_size=5, vad_filter=True,
                           initial_prompt=None, hotwords=None):
                self.kwargs = dict(language=language, initial_prompt=initial_prompt, hotwords=hotwords)
                return iter([]), type("Info", (), {"language_probability": 1.0, "duration": 1.0})()

        def make_engine(**overrides):
            # Model indirmeden STTEngine kur
            engine = STTEngine.__new__(STTEngine)
            engine.model_size, engine.language = "fake", "tr"
            engine.device, engine.compute_type = "cpu", "int8"
            engine.use_domain_prompt = overrides.get("use_domain_prompt", True)
            engine.initial_prompt, engine.hotwords = DOMAIN_INITIAL_PROMPT, DOMAIN_HOTWORDS
            engine.model = FakeModel()
            engine.supports_hotwords = overrides.get("supports_hotwords", True)
            return engine

        self.assertIn("mavi", DOMAIN_INITIAL_PROMPT)
        self.assertIn("sağına", DOMAIN_INITIAL_PROMPT)

        engine = make_engine()
        engine.transcribe(np.ones(1600, dtype=np.float32))
        self.assertEqual(engine.model.kwargs["initial_prompt"], DOMAIN_INITIAL_PROMPT)
        self.assertEqual(engine.model.kwargs["hotwords"], DOMAIN_HOTWORDS)

        # Çağrı bazında kapatma
        engine.transcribe(np.ones(1600, dtype=np.float32), use_domain_prompt=False)
        self.assertIsNone(engine.model.kwargs["initial_prompt"])
        self.assertIsNone(engine.model.kwargs["hotwords"])

        # Motor bazında kapatma
        engine = make_engine(use_domain_prompt=False)
        engine.transcribe(np.ones(1600, dtype=np.float32))
        self.assertIsNone(engine.model.kwargs["initial_prompt"])

        # hotwords desteklemeyen sürüm: yalnızca initial_prompt
        engine = make_engine(supports_hotwords=False)
        engine.transcribe(np.ones(1600, dtype=np.float32))
        self.assertEqual(engine.model.kwargs["initial_prompt"], DOMAIN_INITIAL_PROMPT)
        self.assertIsNone(engine.model.kwargs["hotwords"])

    def test_installed_faster_whisper_supports_hotwords_detection(self):
        from faster_whisper import WhisperModel
        from mahut_kol.speech.stt_engine import _supports_transcribe_kwarg

        self.assertTrue(_supports_transcribe_kwarg(WhisperModel, "initial_prompt"))
        self.assertFalse(_supports_transcribe_kwarg(WhisperModel, "olmayan_parametre"))


if __name__ == "__main__":
    unittest.main()
