"""Common interface so more models (Nemotron, Parakeet, Qwen3-ASR) can be added later."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import BinaryIO, Callable, List, Optional, Union

import numpy as np

AudioInput = Union[str, BinaryIO, np.ndarray]  # path, file-like, or 16 kHz mono float32


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Result:
    text: str
    segments: List[Segment] = field(default_factory=list)
    language: Optional[str] = None
    language_probability: Optional[float] = None
    audio_seconds: float = 0.0
    elapsed: float = 0.0

    @property
    def rtf(self) -> float:
        """Real-time factor: processing time / audio time. Below 1.0 = faster than real time."""
        return self.elapsed / self.audio_seconds if self.audio_seconds else 0.0


class STTEngine(ABC):
    name: str = "base"

    @abstractmethod
    def transcribe(
        self,
        audio: AudioInput,
        *,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        on_segment: Optional[Callable[[Segment], None]] = None,
    ) -> Result:
        ...
