"""
Birim testleri: Ses Kayıt ve Dinleme Modülü (AudioRecorder).
"""

import os
import tempfile
import unittest
import wave
import numpy as np

from mahut_kol.speech.audio_recorder import AudioRecorder, compute_rms


class TestAudioRecorder(unittest.TestCase):
    def setUp(self):
        self.recorder = AudioRecorder(sample_rate=16000, channels=1)

    def test_compute_rms(self):
        """RMS enerji hesaplama fonksiyonunu doğrula."""
        silence = np.zeros(1600, dtype=np.float32)
        self.assertAlmostEqual(compute_rms(silence), 0.0, places=5)

        # 0.5 genlikli kare dalga
        signal = np.ones(1600, dtype=np.float32) * 0.5
        self.assertAlmostEqual(compute_rms(signal), 0.5, places=5)

    def test_save_and_read_wav(self):
        """WAV dosyası oluşturma ve geri okuma formatını (16 kHz, Mono) doğrula."""
        duration = 0.5  # 0.5 saniye
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        # 440 Hz sinüs dalgası
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.recorder.save_wav(audio, tmp_path)
            self.assertTrue(os.path.exists(tmp_path))

            # wave modülü ile özellikleri doğrula
            with wave.open(tmp_path, "rb") as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)  # 16-bit
                self.assertEqual(wf.getframerate(), 16000)
                self.assertEqual(wf.getnframes(), len(audio))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_simulate_vad_detection(self):
        """VAD durum makinesinin simüle edilmiş ses dizisinde sessizlik/konuşmayı tespitini doğrula."""
        sr = 16000
        chunk_size = 512

        # 0.2 saniye sessizlik, 0.4 saniye ses (konuşma), 0.3 saniye sessizlik
        silence_chunk = np.zeros(chunk_size, dtype=np.float32)
        speech_chunk = np.random.uniform(-0.3, 0.3, size=chunk_size).astype(np.float32)

        # compute_rms eşik doğrulaması
        self.assertTrue(compute_rms(silence_chunk) < 0.01)
        self.assertTrue(compute_rms(speech_chunk) > 0.05)

        # process_chunk_vad fonksiyonu testi
        state = self.recorder.create_vad_state(
            energy_threshold=0.02,
            silence_duration_chunks=5
        )

        # Başlangıçta konuşma yok
        self.assertFalse(state["is_speaking"])
        self.assertFalse(state["is_finished"])

        # Sessizlik chunkları besle
        for _ in range(3):
            self.recorder.update_vad_state(state, silence_chunk)
        self.assertFalse(state["is_speaking"])

        # Konuşma chunkları besle
        for _ in range(5):
            self.recorder.update_vad_state(state, speech_chunk)
        self.assertTrue(state["is_speaking"])
        self.assertFalse(state["is_finished"])

        # Konuşma sonrası sessizlik chunkları besle (silence_duration_chunks kadar)
        for _ in range(5):
            self.recorder.update_vad_state(state, silence_chunk)

        # Sessizlik süresi dolduğunda bitti olarak işaretlenmeli
        self.assertTrue(state["is_finished"])


if __name__ == "__main__":
    unittest.main()
