"""
Ses Kayıt ve Dinleme Modülü (AudioRecorder).

Mikrofondan 16 kHz Mono ses verisi yakalar.
Push-to-Talk (Bas-Konuş) ve Enerji Tabanlı VAD (Sessizlik Tespiti) modlarını destekler.
PortAudio kütüphanesini otomatik tespit edip dinamik olarak bağlar.
"""

from __future__ import annotations

import ctypes.util
import os
import sys
import threading
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

# PortAudio dinamik kütüphanesini ~/.local/lib ve diğer yollardan bulma kancası
def _ensure_portaudio_loaded():
    orig_find = ctypes.util.find_library

    def custom_find_library(name: str) -> Optional[str]:
        res = orig_find(name)
        if res is not None:
            return res
        if name in ("portaudio", "libportaudio"):
            candidates = [
                os.path.expanduser("~/.local/lib/libportaudio.so.2"),
                os.path.expanduser("~/.local/lib/libportaudio.so"),
                "/usr/local/lib/libportaudio.so.2",
                "/usr/lib/x86_64-linux-gnu/libportaudio.so.2",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
        return None

    ctypes.util.find_library = custom_find_library

_ensure_portaudio_loaded()

try:
    import sounddevice as sd
except Exception as e:
    sd = None  # type: ignore

import numpy as np


def compute_rms(audio_chunk: np.ndarray) -> float:
    """Verilen ses diliminin karesel ortalama (RMS) enerjisini hesaplar."""
    if audio_chunk is None or audio_chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_chunk, dtype=np.float64))))


class AudioRecorder:
    """16 kHz Mono ses yakalama sınıfı."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size

    def check_audio_device(self) -> bool:
        """Mikrofon ve ses aygıtlarının erişilebilirliğini doğrular."""
        if sd is None:
            return False
        try:
            devices = sd.query_devices()
            return len(devices) > 0
        except Exception:
            return False

    def save_wav(self, audio_data: np.ndarray, file_path: str) -> None:
        """
        NumPy float32 veya int16 ses dizisini 16-bit PCM WAV dosyası olarak kaydeder.
        """
        # [-1.0, 1.0] float32 veriyi 16-bit PCM int16'ya dönüştür
        if audio_data.dtype in (np.float32, np.float64):
            # Kırpma (clipping) ve int16 ölçekleme
            clipped = np.clip(audio_data, -1.0, 1.0)
            int_data = (clipped * 32767.0).astype(np.int16)
        else:
            int_data = audio_data.astype(np.int16)

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 byte = 16 bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(int_data.tobytes())

    def record_fixed_duration(self, duration_seconds: float = 3.0) -> np.ndarray:
        """Belirtilen süre boyunca mikrofondan kayıt yapar."""
        if sd is None:
            raise RuntimeError("sounddevice modülü yüklenemedi. PortAudio kurulu olduğundan emin olun.")

        frames = int(duration_seconds * self.sample_rate)
        recording = sd.rec(
            frames,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocking=True,
        )
        return recording.flatten()

    def record_push_to_talk(
        self,
        prompt_msg: str = "Konuşmaya başlamak için [ENTER]'a basın, bitirmek için tekrar [ENTER]'a basın..."
    ) -> np.ndarray:
        """
        Bas-Konuş modu: İlk Enter ile kayda başlar, ikinci Enter ile kaydı durdurur.
        """
        if sd is None:
            raise RuntimeError("sounddevice modülü yüklenemedi. PortAudio kurulu olduğundan emin olun.")

        input(f"\n🎙️  {prompt_msg}")
        print("🔴 Kayıt yapılıyor... (Konuşmanız bitince tekrar ENTER'a basın)")

        recorded_chunks: List[np.ndarray] = []
        stop_event = threading.Event()

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass
            recorded_chunks.append(indata.copy())

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=audio_callback,
        )

        with stream:
            # Arka planda kullanıcı ikinci kez Enter'a basana kadar bekle
            input()
            stop_event.set()

        print("⏹️  Kayıt durduruldu.")
        if not recorded_chunks:
            return np.zeros(0, dtype=np.float32)

        audio_data = np.concatenate(recorded_chunks, axis=0).flatten()
        return audio_data

    def create_vad_state(
        self,
        energy_threshold: float = 0.02,
        silence_duration_chunks: int = 15,
    ) -> Dict[str, Any]:
        """VAD durum makinesi başlangıç nesnesi oluşturur."""
        return {
            "energy_threshold": energy_threshold,
            "silence_duration_chunks": silence_duration_chunks,
            "is_speaking": False,
            "silent_chunks_count": 0,
            "recorded_chunks": [],
            "is_finished": False,
        }

    def update_vad_state(self, state: Dict[str, Any], chunk: np.ndarray) -> None:
        """Tek bir ses chunk'ı için VAD durumunu günceller."""
        energy = compute_rms(chunk)
        threshold = state["energy_threshold"]

        if not state["is_speaking"]:
            if energy >= threshold:
                state["is_speaking"] = True
                state["recorded_chunks"].append(chunk)
                state["silent_chunks_count"] = 0
        else:
            state["recorded_chunks"].append(chunk)
            if energy < threshold:
                state["silent_chunks_count"] += 1
                if state["silent_chunks_count"] >= state["silence_duration_chunks"]:
                    state["is_finished"] = True
            else:
                state["silent_chunks_count"] = 0

    def record_vad(
        self,
        energy_threshold: float = 0.02,
        silence_duration: float = 1.0,
        max_duration: float = 10.0,
        prompt_msg: str = "Dinleniyor... (Konuşmaya başladığınızda kayıt başlar, susunca durur)"
    ) -> np.ndarray:
        """
        Sessizlik ve Enerji Tabanlı Kayıt (VAD):
        Ses eşiği aşıldığında kayda başlar, silence_duration kadar sessizlik olunca otomatik durur.
        """
        if sd is None:
            raise RuntimeError("sounddevice modülü yüklenemedi. PortAudio kurulu olduğundan emin olun.")

        print(f"\n🎧 {prompt_msg}")
        chunks_per_second = self.sample_rate / self.chunk_size
        silence_chunks_limit = int(silence_duration * chunks_per_second)
        max_total_chunks = int(max_duration * chunks_per_second)

        state = self.create_vad_state(
            energy_threshold=energy_threshold,
            silence_duration_chunks=silence_chunks_limit,
        )

        def audio_callback(indata, frames, time_info, status):
            chunk = indata.copy().flatten()
            self.update_vad_state(state, chunk)

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=audio_callback,
        )

        with stream:
            total_chunks = 0
            while not state["is_finished"] and total_chunks < max_total_chunks:
                time.sleep(0.05)
                total_chunks += 1

        print("⏹️  VAD: Konuşma tamamlandı.")
        if not state["recorded_chunks"]:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(state["recorded_chunks"], axis=0).flatten()
