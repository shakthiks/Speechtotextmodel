import io, wave
import numpy as np
from stt.audio import SR, load_audio, split_chunks


def _wav_bytes(x, sr):
    b = io.BytesIO(); w = wave.open(b, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes((x * 32767).astype(np.int16).tobytes()); w.close(); return b.getvalue()


def test_load_audio_resamples_to_16k_mono():
    sr = 44100
    x = (np.sin(2 * np.pi * 300 * np.arange(sr * 2) / sr) * 0.3).astype(np.float32)
    a = load_audio(io.BytesIO(_wav_bytes(x, sr)))
    assert a.dtype == np.float32 and abs(len(a) - 2 * SR) < 200


def test_split_cuts_in_silence_and_covers_everything():
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(SR * 75) * 0.2).astype(np.float32)
    audio[int(27.5 * SR): int(28.0 * SR)] = 0  # a quiet gap just before the 30 s limit
    chunks = split_chunks(audio, max_s=30)
    assert chunks[0][0] == 0 and chunks[-1][1] == len(audio)
    assert all(b - a <= 30 * SR for a, b in chunks)
    assert all(chunks[i][1] == chunks[i + 1][0] for i in range(len(chunks) - 1))
    assert int(27.5 * SR) <= chunks[0][1] <= int(28.0 * SR)  # cut landed inside the silence


def test_short_audio_single_chunk():
    assert split_chunks(np.zeros(SR * 5, np.float32)) == [(0, SR * 5)]
