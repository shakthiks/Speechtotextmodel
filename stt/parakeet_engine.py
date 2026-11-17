"""NVIDIA Parakeet TDT 0.6B v3 (25 European languages) on CPU via sherpa-onnx (int8 ONNX).

The int8 model (~490 MB download) is fetched once from the sherpa-onnx GitHub release
and cached in ~/.cache/streamlit-stt (override with the STT_MODEL_DIR environment variable).
"""
from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

from .audio import SR, load_audio, split_chunks
from .base import AudioInput, Result, Segment, STTEngine

MODEL_NAME = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{MODEL_NAME}.tar.bz2"
FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")


def model_dir() -> Path:
    base = Path(os.environ.get("STT_MODEL_DIR", Path.home() / ".cache" / "streamlit-stt"))
    return base / MODEL_NAME


def ensure_model(dest: Optional[Path] = None) -> Path:
    """Download + extract the model on first use. Returns the folder holding the files."""
    dest = dest or model_dir()
    if all((dest / f).exists() for f in FILES):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive = Path(tmp) / "model.tar.bz2"
        with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(archive, "wb") as out:
            shutil.copyfileobj(r, out, 1024 * 1024)
        with tarfile.open(archive, "r:bz2") as tar:
            for m in tar:  # extract only the files we need, by base name (no path traversal)
                name = os.path.basename(m.name)
                if m.isfile() and name in FILES:
                    with tar.extractfile(m) as src, open(dest / name, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1024 * 1024)
    missing = [f for f in FILES if not (dest / f).exists()]
    if missing:
        raise RuntimeError(f"Model download incomplete, missing: {missing}")
    return dest


class ParakeetEngine(STTEngine):
    name = "parakeet-tdt-0.6b-v3"

    def __init__(self, cpu_threads: int = 0):
        import sherpa_onnx

        d = ensure_model()
        self._rec = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(d / "encoder.int8.onnx"),
            decoder=str(d / "decoder.int8.onnx"),
            joiner=str(d / "joiner.int8.onnx"),
            tokens=str(d / "tokens.txt"),
            num_threads=cpu_threads or os.cpu_count() or 2,
            sample_rate=SR,
            feature_dim=128,
            decoding_method="greedy_search",
            model_type="nemo_transducer",
        )
        self._lock = threading.Lock()  # one decode at a time (file mode and live mode share this engine)

    def _decode(self, samples) -> str:
        with self._lock:
            s = self._rec.create_stream()
            s.accept_waveform(SR, samples)
            self._rec.decode_stream(s)
            return s.result.text.strip()

    def transcribe(
        self,
        audio: AudioInput,
        *,
        language: Optional[str] = None,   # ignored: Parakeet v3 detects the language itself
        beam_size: int = 5,               # ignored: greedy decoding
        vad_filter: bool = True,          # ignored: long audio is cut at the quietest points instead
        on_segment: Optional[Callable[[Segment], None]] = None,
    ) -> Result:
        samples = load_audio(audio)
        t0 = time.perf_counter()
        segments = []
        for a, b in split_chunks(samples):
            if b - a < 0.1 * SR:
                continue
            text = self._decode(samples[a:b])
            if text:
                seg = Segment(a / SR, b / SR, text)
                segments.append(seg)
                if on_segment:
                    on_segment(seg)
        return Result(
            text=" ".join(s.text for s in segments),
            segments=segments,
            audio_seconds=len(samples) / SR,
            elapsed=time.perf_counter() - t0,
        )
