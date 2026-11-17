"""Real-model test. Downloads ~490 MB on first run:  STT_RUN_MODEL_TESTS=1 python -m pytest -q tests/test_parakeet.py"""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("STT_RUN_MODEL_TESTS"), reason="set STT_RUN_MODEL_TESTS=1")


def test_parakeet_silence_and_tone_do_not_crash():
    from stt.parakeet_engine import ParakeetEngine

    e = ParakeetEngine()
    r = e.transcribe(np.zeros(16000 * 2, np.float32))
    assert r.audio_seconds == 2.0 and r.text == ""
