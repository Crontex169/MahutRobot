"""
MahutKol Sesli Komut Orkestrasyon Modülü (VoiceCommander).

AudioRecorder (ses yakalama), STTEngine (yerel konuşma tanıma) ve
CommandParser (Türkçe kural tabanlı robot komut ayrıştırıcı) modüllerini
birleştirerek tek bir uçtan uca arayüz sunar.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from mahut_kol.speech.audio_recorder import AudioRecorder
from mahut_kol.speech.command_parser import CommandParser, turkish_lower
from mahut_kol.speech.stt_engine import STTEngine

logger = logging.getLogger(__name__)


def _ask_yes_no(prompt: str) -> bool:
    """Klavyeden evet/hayır onayı alır (varsayılan: hayır)."""
    try:
        answer = input(f"❓ {prompt} [e/H]: ")
    except EOFError:
        return False
    return turkish_lower(answer).strip() in ("e", "evet", "y", "yes")


def apply_confirmation(command: Dict[str, Any], confirmed: bool) -> Dict[str, Any]:
    """needs_confirmation=True olan komutu kullanıcı cevabına göre günceller."""
    if not command.get("needs_confirmation"):
        return command
    details = command.get("details") or {}
    details["confirmed_by_user"] = confirmed
    command["details"] = details
    if confirmed:
        command["valid"] = True
        command["needs_confirmation"] = False
        command["error"] = None
    else:
        command["error"] = "Kullanıcı bulanık eşleşmeyi onaylamadı"
    return command


class VoiceCommander:
    """Ses kaydı, transkripsiyon ve robot komut çözümleme orkestratörü."""

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: str = "tr",
        stt_engine: Optional[Any] = None,
        recorder: Optional[Any] = None,
        parser: Optional[CommandParser] = None,
    ):
        self.parser = parser or CommandParser()
        self.recorder = recorder or AudioRecorder(sample_rate=16000, channels=1)

        # Eğer stt_engine dışarıdan enjekte edilmediyse oluştur
        if stt_engine is not None:
            self.stt = stt_engine
        else:
            self.stt = STTEngine(
                model_size=model_size,
                device=device,
                compute_type=compute_type,
                language=language,
            )

    def process_text(self, text: str) -> Dict[str, Any]:
        """Verilen ham Türkçe metni doğrudan JSON komutuna dönüştürür."""
        return self.parser.parse(text)

    def process_audio_array(self, audio: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """NumPy ses dizisini transkribe eder ve robot komutuna ayrıştırır."""
        stt_result = self.stt.transcribe(audio)
        transcript = stt_result.get("text", "").strip()
        command = self.parser.parse(transcript)
        command["stt_metadata"] = {
            "duration": stt_result.get("duration", 0.0),
            "device": stt_result.get("device", ""),
            "compute_type": stt_result.get("compute_type", ""),
        }
        if "error" in stt_result:
            command["error"] = f"STT Hatası: {stt_result['error']}"
            command["valid"] = False
            command["stt_metadata"]["error"] = stt_result["error"]
        return transcript, command

    def process_audio_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Ses dosyasını (WAV vb.) okur, transkribe eder ve ayrıştırır."""
        stt_result = self.stt.transcribe(file_path)
        transcript = stt_result.get("text", "").strip()
        command = self.parser.parse(transcript)
        command["stt_metadata"] = {
            "duration": stt_result.get("duration", 0.0),
            "device": stt_result.get("device", ""),
            "compute_type": stt_result.get("compute_type", ""),
        }
        if "error" in stt_result:
            command["error"] = f"STT Hatası: {stt_result['error']}"
            command["valid"] = False
            command["stt_metadata"]["error"] = stt_result["error"]
        return transcript, command

    def listen_and_parse(
        self,
        mode: str = "push_to_talk",
        **recorder_kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Mikrofondan belirtilen modda ses kaydeder ve komutu ayrıştırır.

        Modlar:
        - 'push_to_talk': Kullanıcı Enter'a bastığında kayıt başlar, tekrar bastığında durur.
        - 'vad': Enerji tabanlı ses aktivitesi tespiti ile otomatik başlar ve durur.
        - 'fixed': Sabit süre (duration_seconds) kayıt yapar.
        """
        if mode == "push_to_talk":
            audio = self.recorder.record_push_to_talk(**recorder_kwargs)
        elif mode == "vad":
            audio = self.recorder.record_vad(**recorder_kwargs)
        elif mode == "fixed":
            audio = self.recorder.record_fixed_duration(**recorder_kwargs)
        else:
            raise ValueError(f"Bilinmeyen kayıt modu: '{mode}'. 'push_to_talk', 'vad' veya 'fixed' olmalı.")

        return self.process_audio_array(audio)

    def run_interactive_loop(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        mode: str = "push_to_talk",
        confirm_fn: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """
        Kullanıcı 'çıkış' veya 'bitir' diyene veya Ctrl+C yapana kadar kesintisiz dinleme döngüsü.

        Bulanık eşleşme düşük güvenliyse (needs_confirmation=True) kullanıcıya
        "Mavi küp mü demek istediniz?" sorusu sorulur; confirm_fn verilmezse klavyeden
        e/H cevabı beklenir. Onaylanmayan komut callback'e valid=False olarak gider.
        """
        confirm = confirm_fn or _ask_yes_no
        print("\n" + "=" * 60)
        print("🤖 MahutKol Sesli Komut Dinleme Döngüsü Başlatıldı")
        print("💡 Çıkmak için 'çıkış', 'bitir' deyin veya Ctrl+C tuşlayın.")
        print("=" * 60)

        try:
            while True:
                transcript, command = self.listen_and_parse(mode=mode)
                print(f"\n📝 Algılanan Metin: \"{transcript}\"")

                norm_exit = turkish_lower(transcript).strip()
                if norm_exit in ("çıkış", "cikis", "bitir", "kapat", "exit", "quit"):
                    print("👋 Sesli komut dinleyici kapatılıyor...")
                    break

                if command.get("needs_confirmation"):
                    question = command.get("confirmation_prompt") or "Komutu onaylıyor musunuz?"
                    print(f"🤔 Emin değilim: {question}")
                    command = apply_confirmation(command, bool(confirm(question)))

                if command["valid"]:
                    print(f"✅ Geçerli Komut: {command['action']} | Nesne: {command['manipulated_object']} | Hedef: {command['target_slot']}")
                else:
                    print(f"⚠️  Anlaşılamadı: {command.get('error', 'Geçersiz komut')}")

                if callback:
                    callback(command)

        except KeyboardInterrupt:
            print("\n🛑 Kullanıcı tarafından durduruldu.")
