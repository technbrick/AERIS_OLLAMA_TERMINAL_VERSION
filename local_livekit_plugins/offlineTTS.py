from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
import wave
import numpy as np
import sounddevice as sd
from typing import TYPE_CHECKING

from livekit.agents import tts, APIConnectOptions

if TYPE_CHECKING:
    from livekit.agents.tts.tts import AudioEmitter

__all__ = ["PiperTTS"]

logger = logging.getLogger(__name__)


class _PiperChunkedStream(tts.ChunkedStream):
    """Internal ChunkedStream implementation for Piper TTS."""

    def __init__(self, *, tts_plugin: PiperTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts_plugin, input_text=input_text, conn_options=conn_options)
        self._piper_tts = tts_plugin

    async def _run(self, emitter: AudioEmitter) -> None:
        emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=self._piper_tts.sample_rate,
            num_channels=self._piper_tts.num_channels,
            mime_type="audio/pcm",
        )

        start_time = time.perf_counter()
        loop = asyncio.get_running_loop()
        audio_bytes = await loop.run_in_executor(None, self._synthesize_blocking, self._input_text)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.debug(f"TTS latency: {elapsed_ms:.0f}ms for {len(self._input_text)} chars")

        emitter.push(audio_bytes)

    def _synthesize_blocking(self, text: str) -> bytes:
        from piper.config import SynthesisConfig

        syn_config = SynthesisConfig(
            length_scale=1.0 / self._piper_tts.speed,
            noise_scale=self._piper_tts.noise_scale,
            noise_w_scale=self._piper_tts.noise_w,
            volume=self._piper_tts.volume,
        )

        # Synthesize to WAV in memory
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wav_file:
            self._piper_tts.voice.synthesize_wav(
                text,
                wav_file,
                syn_config=syn_config,
                set_wav_format=True
            )

        # Extract raw PCM frames
        wav_io.seek(0)
        with wave.open(wav_io, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())

        return frames


class PiperTTS(tts.TTS):
    """LiveKit TTS plugin using Piper with a local speak() method."""

    def __init__(self, model_path: str, use_cuda: bool = False, speed: float = 1.0,
                 volume: float = 1.0, noise_scale: float = 0.667, noise_w: float = 0.8) -> None:
        from piper.voice import PiperVoice

        super().__init__(capabilities=tts.TTSCapabilities(streaming=False),
                         sample_rate=22050,
                         num_channels=1)

        self.speed = speed
        self.volume = volume
        self.noise_scale = noise_scale
        self.noise_w = noise_w

        logger.info(f"Loading Piper voice: {model_path} (CUDA: {use_cuda})")
        self.voice = PiperVoice.load(model_path, use_cuda=use_cuda)
        logger.info(f"Piper ready - speed={speed}, volume={volume}")

    def synthesize(self, text: str, *, conn_options: APIConnectOptions | None = None) -> tts.ChunkedStream:
        if conn_options is None:
            conn_options = APIConnectOptions()
        logger.debug(f"Synthesizing ({len(text)} chars): {text[:50]}...")
        return _PiperChunkedStream(tts_plugin=self, input_text=text, conn_options=conn_options)

    def speak(self, text: str) -> None:
        """Synthesize and play text locally via sounddevice."""
        # Use the internal blocking synth
        chunked = _PiperChunkedStream(tts_plugin=self, input_text=text, conn_options=APIConnectOptions())
        audio_bytes = chunked._synthesize_blocking(text)

        # Convert PCM bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # Play via sounddevice
        sd.play(audio_array, samplerate=self.sample_rate)
        sd.wait()
