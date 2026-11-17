"""Live (streaming) transcription.

Audio frames arrive from the browser (streamlit-webrtc). We split the stream into
utterances with a simple energy-based VAD, and a background worker thread:
  * emits a quick "partial" transcript every few seconds while someone is speaking
  * emits a "final" transcript when they pause (or the utterance gets too long)
Results are put on `events` and the Streamlit script drains that queue.
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Optional

import numpy as np

SR = 16_000
MIN_VOICED_S = 0.25   # ignore blips shorter than this
PREROLL_S = 0.3       # audio kept from before speech starts so first word isn't clipped


class LiveTranscriber:
    def __init__(
        self,
        engine,
        language: Optional[str] = None,
        beam_size: int = 1,
        silence_ms: int = 700,
        max_utterance_s: float = 20.0,
        partial_every_s: float = 1.5,
        energy_threshold: float = 0.008,
        enable_partials: bool = True,
    ):
        # all of these can be changed while running
        self.engine = engine
        self.language = language
        self.beam_size = beam_size
        self.silence_ms = silence_ms
        self.max_utterance_s = max_utterance_s
        self.partial_every_s = partial_every_s
        self.energy_threshold = energy_threshold
        self.enable_partials = enable_partials

        self.events: "queue.Queue[dict]" = queue.Queue()
        self._lock = threading.Lock()        # protects the audio buffer
        self._infer_lock = threading.Lock()  # one model call at a time
        self._resampler = None
        # diagnostics shown in the UI
        self.frames = 0       # audio frames received from the browser (0 = nothing is arriving)
        self.level = 0.0      # smoothed mic loudness (RMS)
        self._preroll: list = []
        self._pre_n = 0
        self._reset_utterance()

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True, name="live-stt")
        self._thread.start()

    @property
    def speaking(self) -> bool:
        return self._speaking

    # ---------------------------------------------------------------- input
    def push_frame(self, frame) -> None:
        """Accept an av.AudioFrame from streamlit-webrtc (any rate/channels)."""
        import av

        if self._resampler is None:
            self._resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
        out = self._resampler.resample(frame)
        for f in out if isinstance(out, list) else [out]:
            if f is None:
                continue
            pcm = f.to_ndarray().reshape(-1).astype(np.float32) / 32768.0
            self.push(pcm)

    def push(self, pcm: np.ndarray) -> None:
        """Accept 16 kHz mono float32 audio in [-1, 1]."""
        if len(pcm) == 0:
            return
        rms = float(np.sqrt(np.mean(pcm * pcm)))
        self.frames += 1
        self.level = max(rms, self.level * 0.85)   # fast attack, slow decay: easy to read as a meter
        voiced = rms >= self.energy_threshold
        with self._lock:
            if voiced:
                if not self._speaking:
                    self._speaking = True
                    self._buf = list(self._preroll)
                    self._n = self._pre_n
                    self._preroll, self._pre_n = [], 0
                    self._last_partial_t = time.monotonic()
                    self._last_partial_n = 0
                self._buf.append(pcm)
                self._n += len(pcm)
                self._voiced += len(pcm)
                self._silence = 0
            elif self._speaking:
                self._buf.append(pcm)
                self._n += len(pcm)
                self._silence += len(pcm)
            else:
                self._preroll.append(pcm)
                self._pre_n += len(pcm)
                limit = int(PREROLL_S * SR)
                while self._pre_n > limit and len(self._preroll) > 1:
                    self._pre_n -= len(self._preroll.pop(0))

    # ------------------------------------------------------------ lifecycle
    def flush(self) -> None:
        """Transcribe whatever is left (call when the user presses STOP)."""
        with self._lock:
            audio, voiced = self._take_buffer()
        if audio is not None and voiced >= MIN_VOICED_S * SR:
            self._run("final", audio)

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------- internals
    def _reset_utterance(self) -> None:
        self._speaking = False
        self._buf: list = []
        self._n = 0
        self._voiced = 0
        self._silence = 0
        self._last_partial_t = 0.0
        self._last_partial_n = 0

    def _take_buffer(self):
        if not self._buf:
            self._reset_utterance()
            return None, 0
        audio, voiced = np.concatenate(self._buf), self._voiced
        self._reset_utterance()
        return audio, voiced

    def _next_job(self):
        with self._lock:
            if not self._speaking:
                return None
            silent = self._silence >= int(self.silence_ms / 1000 * SR)
            too_long = self._n >= int(self.max_utterance_s * SR)
            if silent or too_long:
                audio, voiced = self._take_buffer()
                if audio is None or voiced < MIN_VOICED_S * SR:
                    return None
                return "final", audio
            now = time.monotonic()
            if (
                self.enable_partials
                and self._voiced >= MIN_VOICED_S * SR
                and now - self._last_partial_t >= self.partial_every_s
                and self._n - self._last_partial_n >= 0.5 * SR
            ):
                self._last_partial_t, self._last_partial_n = now, self._n
                return "partial", np.concatenate(self._buf)
        return None

    def _worker(self) -> None:
        while not self._stop.wait(0.15):
            job = self._next_job()
            if job:
                self._run(*job)

    def _run(self, kind: str, audio: np.ndarray) -> None:
        try:
            with self._infer_lock:
                res = self.engine.transcribe(
                    audio.astype(np.float32),
                    language=self.language,
                    beam_size=self.beam_size if kind == "final" else 1,
                    vad_filter=True,
                )
            if res.text:
                self.events.put(
                    {
                        "kind": kind,
                        "text": res.text,
                        "language": res.language,
                        "audio_s": len(audio) / SR,
                        "elapsed": res.elapsed,
                    }
                )
        except Exception as e:  # keep the worker alive
            self.events.put({"kind": "error", "text": str(e)})