from __future__ import annotations

import logging
import time
from typing import Literal, Optional

import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# -----------------------------
# Type Aliases
# -----------------------------
ModelSize = Literal[
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v2", "large-v3"
]

Device = Literal["cuda", "cpu", "auto"]

ComputeType = Literal[
    "float16",
    "float32",
    "int8",
    "int8_float16",
    "int8_float32"
]


# -------------------------------------------------
# FULL OFFLINE STT (NO LIVEKIT)
# -------------------------------------------------
class FasterWhisperSTT:
    """
    Fully offline Faster-Whisper wrapper.
    Designed for standalone usage (no LiveKit).
    """

    def __init__(
        self,
        model_size: ModelSize = "base",
        device: Device = "cpu",
        compute_type: ComputeType = "int8",
        language: Optional[str] = "en",
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> None:

        self._language = language
        self._beam_size = beam_size
        self._vad_filter = vad_filter

        logger.info(
            f"Loading FasterWhisper model: {model_size} "
            f"on {device} ({compute_type})"
        )

        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        logger.info(
            f"FasterWhisper ready - "
            f"language={language}, beam_size={beam_size}"
        )

    # -------------------------------------------------
    # Transcribe from WAV file
    # -------------------------------------------------
    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe a WAV file and return text.
        """

        start_time = time.perf_counter()

        segments, info = self._model.transcribe(
            audio_path,
            beam_size=self._beam_size,
            best_of=self._beam_size,
            temperature=0.0,
            vad_filter=self._vad_filter,
            language=self._language,
        )

        text = "".join(segment.text for segment in segments).strip()

        elapsed = (time.perf_counter() - start_time) * 1000

        if text:
            logger.debug(
                f"Transcribed ({info.language}, "
                f"{info.duration:.1f}s audio) → {elapsed:.0f}ms"
            )

        return text

    # -------------------------------------------------
    # Transcribe from NumPy audio (optional)
    # -------------------------------------------------
    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe raw numpy float32 audio array.
        """

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        start_time = time.perf_counter()

        segments, info = self._model.transcribe(
            audio,
            beam_size=self._beam_size,
            best_of=self._beam_size,
            temperature=0.0,
            vad_filter=self._vad_filter,
            language=self._language,
        )

        text = "".join(segment.text for segment in segments).strip()

        elapsed = (time.perf_counter() - start_time) * 1000

        logger.debug(
            f"Array transcription ({info.duration:.1f}s) → {elapsed:.0f}ms"
        )

        return text
