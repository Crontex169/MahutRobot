"""
Entegrasyon testleri: VoiceCommander Orkestrasyon Sınıfı.
"""

import unittest
import numpy as np

from mahut_kol.speech import (
    AudioRecorder,
    CommandParser,
    STTEngine,
    VoiceCommander,
    turkish_lower,
)


class DummySTT:
    """Testler için sahte STT motoru."""
    def __init__(self, return_text="Kırmızı küpü al yeşil silindirin sağına koy"):
        self.return_text = return_text

    def transcribe(self, audio, language="tr"):
        return {
            "text": self.return_text,
            "language": language,
            "duration": 1.0,
        }


class DummyRecorder:
    """Testler için sahte ses kaydedici."""
    def record_push_to_talk(self, *args, **kwargs):
        return np.zeros(16000, dtype=np.float32)

    def record_vad(self, *args, **kwargs):
        return np.zeros(16000, dtype=np.float32)


class TestVoiceCommander(unittest.TestCase):
    def test_package_exports(self):
        """Paket başlığından tüm temel sınıfların aktarıldığını doğrula."""
        self.assertTrue(callable(VoiceCommander))
        self.assertTrue(callable(CommandParser))
        self.assertTrue(callable(STTEngine))
        self.assertTrue(callable(AudioRecorder))
        self.assertTrue(callable(turkish_lower))

    def test_process_text(self):
        """Doğrudan metin iletiminin doğru çözümlendiğini test et."""
        commander = VoiceCommander(
            stt_engine=DummySTT(),
            recorder=DummyRecorder(),
        )
        res = commander.process_text("Mavi silindiri it")
        self.assertTrue(res["valid"])
        self.assertEqual(cmd_action := res["action"], "push")
        self.assertEqual(res["manipulated_object"], "blue_cylinder")

    def test_listen_and_parse_with_mock(self):
        """Sahte bileşenlerle ses yakalama -> STT -> komut çözümleme pipeline'ını test et."""
        dummy_stt = DummySTT("Kırmızı küpü al yeşil silindirin sağına koy")
        dummy_rec = DummyRecorder()
        commander = VoiceCommander(stt_engine=dummy_stt, recorder=dummy_rec)

        transcript, cmd = commander.listen_and_parse(mode="push_to_talk")
        self.assertEqual(transcript, "Kırmızı küpü al yeşil silindirin sağına koy")
        self.assertTrue(cmd["valid"])
        self.assertEqual(cmd["action"], "pick_and_place")
        self.assertEqual(cmd["manipulated_object"], "red_cube")
        self.assertEqual(cmd["reference_object"], "green_cylinder")
        self.assertEqual(cmd["relation"], "RIGHT")

        # VAD modunu da test et
        transcript, cmd = commander.listen_and_parse(mode="vad")
        self.assertEqual(transcript, "Kırmızı küpü al yeşil silindirin sağına koy")
        self.assertTrue(cmd["valid"])


if __name__ == "__main__":
    unittest.main()
