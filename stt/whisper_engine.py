from __future__ import annotations

import time
from typing import Callable, Optional

from .base import AudioInput, Result, Segment, STTEngine

# code -> display name (Indian languages first, then common ones)
LANGUAGES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "ur": "Urdu",
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "ru": "Russian", "ar": "Arabic", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "id": "Indonesian",
}


class WhisperEngine(STTEngine):
    """faster-whisper (CTranslate2) on CPU with int8 quantisation."""

    def __init__(self, model_size: str = "large-v3-turbo", compute_type: str = "int8", cpu_threads: int = 0):
        from faster_whisper import WhisperModel  # imported lazily so the UI loads fast

        self.name = f"whisper-{model_size}"
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,  # 0 = let CTranslate2 decide
            num_workers=1,
        )

    def transcribe(
        self,
        audio: AudioInput,
        *,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[Segment], None]] = None,
    ) -> Result:
        t0 = time.perf_counter()
        segments_iter, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,  # avoids repetition loops on long audio
        )
        segments = []
        for s in segments_iter:  # decoding happens lazily here
            seg = Segment(s.start, s.end, s.text.strip())
            segments.append(seg)
            if on_segment:
                on_segment(seg)
        return Result(
            text=" ".join(s.text for s in segments if s.text).strip(),
            segments=segments,
            language=info.language,
            language_probability=info.language_probability,
            audio_seconds=info.duration,
            elapsed=time.perf_counter() - t0,
        )
