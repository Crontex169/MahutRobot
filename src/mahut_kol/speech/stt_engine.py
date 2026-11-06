"""
Faster-Whisper Yerel Türkçe Speech-to-Text Motoru.

CTranslate2 tabanlı faster-whisper kütüphanesini sarmalar.
CUDA (float16) ve CPU (int8) üzerinde yüksek verimlilikle yerel (offline) çalışır.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


def resolve_device_and_compute_type(
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Kullanılabilir donanım ve en uygun hesaplama tipini belirler.
    CUDA varsa: 'cuda', 'float16'
    CPU için: 'cpu', 'int8'
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = device.lower()

    if compute_type is None:
        if device == "cuda":
            compute_type = "float16"
        else:
            compute_type = "int8"

    return device, compute_type


class STTEngine:
    """Yerel Faster-Whisper konuşma tanıma (STT) motoru."""

    def __init__(
        self,
        model_size: str = "base",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: str = "tr",
        cpu_threads: int = 4,
    ):
        self.model_size = model_size
        self.language = language
        self.cpu_threads = cpu_threads

        resolved_device, resolved_compute = resolve_device_and_compute_type(device, compute_type)
        self.device = resolved_device
        self.compute_type = resolved_compute

        self.model = self._load_model()

    def _load_model(self) -> WhisperModel:
        """Faster-Whisper modelini yükler. CUDA arızalanırsa zarifçe CPU int8 moduna geçer."""
        try:
            logger.info(
                f"Faster-Whisper modeli yükleniyor: {self.model_size} ({self.device}, {self.compute_type})"
            )
            return WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.cpu_threads,
            )
        except Exception as e:
            if self.device == "cuda":
                logger.warning(
                    f"CUDA üzerinde model başlatılamadı ({e}). CPU int8 moduna geçiliyor..."
                )
                self.device = "cpu"
                self.compute_type = "int8"
                return WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=self.cpu_threads,
                )
            raise e

    def transcribe(
        self,
        audio: Union[str, np.ndarray, bytes],
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Dict[str, Any]:
        """
        Girdi sesini (dosya yolu veya 16 kHz float32 numpy dizisi) metne dönüştürür.
        """
        target_lang = language or self.language

        # NumPy dizisi boş ise doğrudan boş sonuç dön
        if isinstance(audio, np.ndarray):
            if audio.size == 0:
                return {
                    "text": "",
                    "language": target_lang,
                    "language_probability": 0.0,
                    "duration": 0.0,
                    "segments": [],
                    "device": self.device,
                    "compute_type": self.compute_type,
                    "model_size": self.model_size,
                }
            # 1D ve float32 formatı sağla
            if audio.ndim > 1:
                audio = audio.flatten()
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

        try:
            segments, info = self.model.transcribe(
                audio,
                language=target_lang,
                beam_size=beam_size,
                vad_filter=vad_filter,
            )

            segment_list = list(segments)
            full_text = " ".join(s.text.strip() for s in segment_list).strip()

            return {
                "text": full_text,
                "language": target_lang,
                "language_probability": getattr(info, "language_probability", 1.0),
                "duration": getattr(info, "duration", 0.0),
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text.strip()}
                    for s in segment_list
                ],
                "device": self.device,
                "compute_type": self.compute_type,
                "model_size": self.model_size,
            }
        except Exception as e:
            logger.error(f"Transkripsiyon hatası: {e}")
            return {
                "text": "",
                "error": str(e),
                "language": target_lang,
                "device": self.device,
                "compute_type": self.compute_type,
                "model_size": self.model_size,
            }
