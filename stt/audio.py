"""Audio helpers shared by engines that don't decode files themselves."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

SR = 16_000


def _as_list(x):
    return x if isinstance(x, list) else ([] if x is None else [x])


def load_audio(src, sr: int = SR) -> np.ndarray:
    """Decode a path / file-like object (wav, mp3, m4a, webm, mp4, ...) to mono float32 at `sr`.
    A numpy array is returned unchanged (assumed to be `sr` mono float32)."""
    if isinstance(src, np.ndarray):
        return src.astype(np.float32, copy=False)
    import av

    if hasattr(src, "seek"):
        src.seek(0)
    parts = []
    with av.open(src, mode="r") as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError("No audio track found in this file.")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
        for frame in container.decode(stream):
            for f in _as_list(resampler.resample(frame)):
                parts.append(f.to_ndarray().reshape(-1))
        for f in _as_list(resampler.resample(None)):  # flush
            parts.append(f.to_ndarray().reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32) / 32768.0


def split_chunks(audio: np.ndarray, sr: int = SR, max_s: float = 15.0, search_s: float = 4.0) -> List[Tuple[int, int]]:
    """Cut long audio into pieces of at most `max_s`, always cutting at the quietest
    moment in the last `search_s` seconds so words are not split in half."""
    n, max_n = len(audio), int(max_s * sr)
    win = int(0.05 * sr)
    chunks, start = [], 0
    while n - start > max_n:
        lo, hi = start + max_n - int(search_s * sr), start + max_n
        seg = audio[lo:hi].astype(np.float64)
        c = np.concatenate([[0.0], np.cumsum(seg * seg)])
        energy = c[win:] - c[:-win]
        cut = lo + int(np.argmin(energy)) + win // 2
        chunks.append((start, cut))
        start = cut
    chunks.append((start, n))
    return chunks
