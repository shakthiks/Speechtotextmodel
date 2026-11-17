import os
import time

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from stt import LANGUAGES, MODELS, LiveTranscriber, create_engine

st.set_page_config(page_title="CPU Speech-to-Text", page_icon="🎙️", layout="wide")
st.title("🎙️ Speech-to-Text on CPU")
st.caption("Record, upload, or stream live. Runs locally on CPU with Parakeet (sherpa-onnx) or Whisper (faster-whisper), no GPU needed.")


@st.cache_resource(show_spinner="Loading model (the first run downloads it, this can take a few minutes)…")
def get_engine(model_label: str, compute_type: str, cpu_threads: int):
    return create_engine(model_label, compute_type, cpu_threads)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Model")
    model_label = st.selectbox("Model", list(MODELS))
    spec = MODELS[model_label]
    if spec.supports_language:
        lang_name = st.selectbox("Language", ["Auto-detect"] + list(LANGUAGES.values()))
        language = None if lang_name == "Auto-detect" else next(k for k, v in LANGUAGES.items() if v == lang_name)
    else:
        language = None
        st.caption(spec.note)

    with st.expander("Advanced"):
        compute_type = st.selectbox("Precision (Whisper)", ["int8", "int8_float32", "float32"], help="int8 is fastest on CPU. Parakeet is always int8.")
        cpu_threads = st.number_input("CPU threads (0 = auto)", 0, 64, 0)
        beam_size = st.slider("Beam size (file mode)", 1, 8, 5, help="Whisper only. Higher = more accurate, slower.")
        vad_filter = st.checkbox("Skip silence (VAD)", True, help="Whisper only.")

    st.header("Live settings")
    silence_ms = st.slider("Pause that ends a sentence (ms)", 300, 2000, 700, 100)
    energy = st.slider("Mic sensitivity threshold", 0.001, 0.05, 0.008, 0.001, format="%.3f", help="Lower = picks up quieter speech. Watch the Mic level meter in the Live tab and set this just below your normal speaking level.")
    partials = st.checkbox("Show partial text while speaking", True, help="Uses extra CPU. Turn off on slow machines.")
    partial_every = st.slider("Partial update every (s)", 0.5, 5.0, 1.5, 0.5)

engine = get_engine(model_label, compute_type, int(cpu_threads))


# ------------------------------------------------------------------ helpers
def srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def to_srt(segments) -> str:
    return "\n".join(
        f"{i}\n{srt_time(s.start)} --> {srt_time(s.end)}\n{s.text}\n" for i, s in enumerate(segments, 1)
    )


def show_result(res) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Detected language", f"{res.language} ({res.language_probability:.0%})" if res.language else "-")
    c2.metric("Audio length", f"{res.audio_seconds:.1f}s")
    c3.metric("Processing time", f"{res.elapsed:.1f}s", f"RTF {res.rtf:.2f} (<1 = faster than real time)", delta_color="off")
    st.text_area("Transcript", res.text, height=200)
    d1, d2 = st.columns(2)
    d1.download_button("Download .txt", res.text, "transcript.txt")
    d2.download_button("Download .srt", to_srt(res.segments), "transcript.srt")


def transcribe_bytes(data: bytes, key: str) -> None:
    import io

    live_box = st.empty()
    parts: list[str] = []

    def on_segment(seg):
        parts.append(seg.text)
        live_box.markdown(" ".join(parts))

    with st.spinner("Transcribing…"):
        st.session_state[key] = engine.transcribe(
            io.BytesIO(data), language=language, beam_size=beam_size, vad_filter=vad_filter, on_segment=on_segment
        )
    live_box.empty()


def rtc_config():
    """Network settings for the live microphone connection (WebRTC).
    - If Cloudflare TURN keys are set (needed on Streamlit Community Cloud), return None so streamlit-webrtc
      fetches TURN credentials itself from CLOUDFLARE_TURN_KEY_ID / CLOUDFLARE_TURN_KEY_API_TOKEN.
    - Otherwise use a free public STUN server (fine on localhost and most home networks)."""
    if os.environ.get("CLOUDFLARE_TURN_KEY_ID") and os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN"):
        return None
    return {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


tab_rec, tab_up, tab_live = st.tabs(["🎤 Record", "📁 Upload", "🔴 Live"])

# ------------------------------------------------------------------- record
with tab_rec:
    audio = st.audio_input("Click the mic, speak, then click stop")
    if audio is not None and st.button("Transcribe recording", type="primary", key="go_rec"):
        transcribe_bytes(audio.getvalue(), "res_rec")
    if "res_rec" in st.session_state:
        show_result(st.session_state["res_rec"])

# ------------------------------------------------------------------- upload
with tab_up:
    up = st.file_uploader("Audio or video file", type=["wav", "mp3", "m4a", "flac", "ogg", "opus", "webm", "mp4"])
    if up is not None:
        st.audio(up)
        if st.button("Transcribe file", type="primary", key="go_up"):
            transcribe_bytes(up.getvalue(), "res_up")
    if "res_up" in st.session_state:
        show_result(st.session_state["res_up"])

# --------------------------------------------------------------------- live
with tab_live:
    st.write("Press **START**, allow the microphone, and speak. Text appears as you talk.")
    if "live" not in st.session_state:
        st.session_state.live = LiveTranscriber(engine)
        st.session_state.finals = []
    live: LiveTranscriber = st.session_state.live
    # apply the latest sidebar settings (works while streaming)
    live.engine, live.language = engine, language
    live.silence_ms, live.energy_threshold = silence_ms, energy
    live.enable_partials, live.partial_every_s = partials, partial_every

    def on_frame(frame):
        live.push_frame(frame)
        return frame

    ctx = webrtc_streamer(
        key="live-stt",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=on_frame,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration=rtc_config(),
    )

    status = st.empty()
    meter = st.empty()
    out = st.empty()
    if st.button("Clear transcript"):
        st.session_state.finals = []

    def render(partial: str = "") -> None:
        text = " ".join(st.session_state.finals)
        out.markdown(text + (f" *{partial}*" if partial else "") or "_Waiting for speech…_")

    partial_text = ""
    render()
    was_playing = False
    frames0, t_start = live.frames, time.time()
    while ctx.state.playing:
        was_playing = True
        got = live.frames - frames0
        if got == 0:
            if time.time() - t_start > 8:
                status.error("No audio is arriving. Check the browser mic permission, the selected microphone, "
                             "and that your network allows WebRTC (a corporate network may block it).")
            else:
                status.warning("🟡 Connecting… wait until this says Listening, then start speaking.")
        else:
            status.info("🔴 Listening. You can speak now.")
        n = min(int(live.level / 0.05 * 20), 20)
        heard = live.level >= live.energy_threshold
        meter.markdown(
            f"Mic level `{'█' * n}{'░' * (20 - n)}` {live.level:.3f} · threshold {live.energy_threshold:.3f} · "
            + ("🟢 speech detected" if heard else "⚪ below threshold (not counted as speech)")
        )
        while not live.events.empty():
            ev = live.events.get()
            if ev["kind"] == "partial":
                partial_text = ev["text"]
            elif ev["kind"] == "final":
                st.session_state.finals.append(ev["text"])
                partial_text = ""
            elif ev["kind"] == "error":
                status.error(ev["text"])
        render(partial_text)
        time.sleep(0.2)

    if was_playing:  # user pressed STOP: transcribe the tail
        live.flush()
        while not live.events.empty():
            ev = live.events.get()
            if ev["kind"] == "final":
                st.session_state.finals.append(ev["text"])
        status.success("Stopped.")
        render()
    if st.session_state.finals:
        st.download_button("Download transcript", " ".join(st.session_state.finals), "live_transcript.txt")