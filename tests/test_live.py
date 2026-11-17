"""Run: python -m pytest -q   (uses a fake engine, no model download needed)"""
import time

import numpy as np

from stt.base import Result
from stt.live import SR, LiveTranscriber


class FakeEngine:
    def transcribe(self, audio, **kw):
        return Result(text=f"heard {len(audio)/SR:.1f}s", audio_seconds=len(audio) / SR)


def feed(lt, seconds, amp):
    chunk = int(0.02 * SR)
    for _ in range(int(seconds / 0.02)):
        lt.push((np.random.randn(chunk) * amp).astype(np.float32))


def drain(lt, timeout=3.0):
    t0, out = time.time(), []
    while time.time() - t0 < timeout:
        while not lt.events.empty():
            out.append(lt.events.get())
        time.sleep(0.05)
    return out


def test_speech_then_pause_gives_final():
    lt = LiveTranscriber(FakeEngine(), silence_ms=400, enable_partials=False)
    feed(lt, 0.2, 0.0005)   # silence (pre-roll)
    feed(lt, 1.5, 0.2)      # speech
    feed(lt, 0.8, 0.0005)   # pause
    ev = drain(lt, 1.5)
    lt.stop()
    finals = [e for e in ev if e["kind"] == "final"]
    assert len(finals) == 1 and finals[0]["audio_s"] > 1.5


def test_partials_while_speaking():
    lt = LiveTranscriber(FakeEngine(), partial_every_s=0.5)
    for _ in range(8):      # 4s of speech in real time
        feed(lt, 0.5, 0.2)
        time.sleep(0.5)
    ev = drain(lt, 0.5)
    lt.stop()
    assert any(e["kind"] == "partial" for e in ev)


def test_noise_blip_ignored():
    lt = LiveTranscriber(FakeEngine(), silence_ms=300)
    feed(lt, 0.06, 0.2)
    feed(lt, 1.0, 0.0005)
    assert not [e for e in drain(lt, 1.0) if e["kind"] == "final"]
    lt.stop()


def test_flush_returns_tail():
    lt = LiveTranscriber(FakeEngine(), enable_partials=False)
    feed(lt, 1.0, 0.2)
    lt.flush()
    ev = drain(lt, 0.3)
    lt.stop()
    assert any(e["kind"] == "final" for e in ev)
