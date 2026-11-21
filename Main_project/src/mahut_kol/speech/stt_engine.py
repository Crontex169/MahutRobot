"""
Faster-Whisper Yerel Türkçe Speech-to-Text Motoru.

CTranslate2 tabanlı faster-whisper kütüphanesini sarmalar.
CUDA (float16) ve CPU (int8) üzerinde yüksek verimlilikle yerel (offline) çalışır.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# --- Domain sözlüğü (Whisper'ı küçük komut kümesine yönlendirmek için) ---
# Whisper'ın initial_prompt'u "önceki konuşma" gibi davranır; kelime listesinden
# çok doğal cümleler daha iyi sonuç verir. 224 token sınırının epey altında tutulur.
DOMAIN_INITIAL_PROMPT = (
    "Robot kol komutları. Kırmızı küpü al, yeşil silindirin sağına koy. "
    "Mavi üçgen prizmayı kaldır, kırmızı küpün soluna bırak. "
    "Yeşil küpü mavi silindirin önüne yerleştir. Mavi küpü kırmızı prizmanın arkasına taşı. "
    "Kırmızı silindiri yeşil üçgenin üstüne koy. Mavi silindiri it. "
    "Renkler: kırmızı, yeşil, mavi. Şekiller: küp, silindir, üçgen prizma. "
    "Yönler: önüne, arkasına, soluna, sağına, üstüne, üzerine."
)

# hotwords destekleyen faster-whisper sürümlerinde (>=1.0.2) ek ipucu olarak verilir.
DOMAIN_HOTWORDS = (
    "kırmızı yeşil mavi küp küpü silindir silindiri silindirin üçgen prizma prizmayı "
    "önüne arkasına soluna sağına üstüne üzerine al koy bırak yerleştir taşı it kaldır tut"
)


def _supports_transcribe_kwarg(model: Any, name: str) -> bool:
    """Yüklü faster-whisper sürümünün transcribe() içinde verilen parametreyi destekleyip desteklemediği."""
    try:
        return name in inspect.signature(model.transcribe).parameters
    except (TypeError, ValueError):
        return False


def resolve_device_and_compute_type(
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Kullanılabilir donanım ve en uygun hesaplama tipini belirler.
    CUDA varsa: 'cuda', 'float16'
    CPU için: 'cpu', 'int8'
    """
    if device is None or device.lower() == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
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
        use_domain_prompt: bool = True,
        initial_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ):
        """
        use_domain_prompt: True ise transcribe() çağrısına robot komut sözlüğü
            (renk/şekil/yön/eylem) initial_prompt ve (destekleniyorsa) hotwords olarak verilir.
        initial_prompt / hotwords: Varsayılan domain metinlerinin yerine kullanılacak özel metinler.
        """
        self.model_size = model_size
        self.language = language
        self.cpu_threads = cpu_threads
        self.use_domain_prompt = use_domain_prompt
        self.initial_prompt = initial_prompt if initial_prompt is not None else DOMAIN_INITIAL_PROMPT
        self.hotwords = hotwords if hotwords is not None else DOMAIN_HOTWORDS

        resolved_device, resolved_compute = resolve_device_and_compute_type(device, compute_type)
        self.device = resolved_device
        self.compute_type = resolved_compute

        self.model = self._load_model()
        self.supports_hotwords = _supports_transcribe_kwarg(self.model, "hotwords")

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

    def _build_transcribe_kwargs(
        self,
        language: str,
        beam_size: int,
        vad_filter: bool,
        use_domain_prompt: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """model.transcribe() parametrelerini hazırlar (domain sözlüğü dahil)."""
        kwargs: Dict[str, Any] = {
            "language": language,
            "beam_size": beam_size,
            "vad_filter": vad_filter,
        }
        enabled = self.use_domain_prompt if use_domain_prompt is None else use_domain_prompt
        if enabled:
            if self.initial_prompt:
                kwargs["initial_prompt"] = self.initial_prompt
            if self.hotwords and getattr(self, "supports_hotwords", False):
                kwargs["hotwords"] = self.hotwords
        return kwargs

    def transcribe(
        self,
        audio: Union[str, np.ndarray, bytes],
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        use_domain_prompt: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Girdi sesini (dosya yolu veya 16 kHz float32 numpy dizisi) metne dönüştürür.

        use_domain_prompt: None ise motorun varsayılanı (self.use_domain_prompt) kullanılır;
            False verilirse bu çağrı için domain sözlüğü kapatılır.
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
                **self._build_transcribe_kwargs(target_lang, beam_size, vad_filter, use_domain_prompt),
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
