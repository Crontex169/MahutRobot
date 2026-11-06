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


if __name__ == "__main__":
    unittest.main()
