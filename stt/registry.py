"""The list of models shown in the UI. To add a model: write an engine, add a ModelSpec here."""
from __future__ import annotations

from dataclasses import dataclass

from .base import STTEngine


@dataclass(frozen=True)
class ModelSpec:
    kind: str                 # "parakeet" | "whisper"
    arg: str = ""             # e.g. whisper model size
    supports_language: bool = False
    note: str = ""


MODELS = {
    "Parakeet TDT 0.6B v3 (fastest on CPU, 25 European languages)": ModelSpec(
        "parakeet",
        note="Detects the language automatically. European languages only (no Hindi/Tamil/etc.).",
    ),
    "Whisper large-v3-turbo (best coverage, ~99 languages)": ModelSpec("whisper", "large-v3-turbo", True),
    "Whisper small (lighter, multilingual)": ModelSpec("whisper", "small", True),
    "Whisper base (fastest Whisper, lowest accuracy)": ModelSpec("whisper", "base", True),
}


def create_engine(label: str, compute_type: str = "int8", cpu_threads: int = 0) -> STTEngine:
    spec = MODELS[label]
    if spec.kind == "parakeet":
        from .parakeet_engine import ParakeetEngine

        return ParakeetEngine(cpu_threads)
    from .whisper_engine import WhisperEngine

    return WhisperEngine(spec.arg, compute_type, cpu_threads)
