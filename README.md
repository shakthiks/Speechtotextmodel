# 🎙️ Speech-to-Text on CPU (Streamlit)

Pick a model, then **record**, **upload**, or **stream live** from your microphone. CPU only, no GPU.

| Model in the dropdown | Engine | Languages | Notes |
|---|---|---|---|
| **Parakeet TDT 0.6B v3** (default) | sherpa-onnx, int8 ONNX | 25 European languages, auto-detected | Fastest on CPU (~0.1-0.15 RTF measured on 2 cores). No Hindi/Tamil etc. |
| Whisper large-v3-turbo / small / base | faster-whisper, int8 | ~99 languages incl. Indian languages | Best language coverage, slower |

| Tab | How it works |
|---|---|
| 🎤 Record | `st.audio_input` records in the browser, then transcribes the clip |
| 📁 Upload | wav, mp3, m4a, flac, ogg, webm, mp4; download `.txt` / `.srt` |
| 🔴 Live | `streamlit-webrtc` streams mic audio → energy VAD splits sentences → partial text while you speak, final text at each pause |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
First use of a model downloads it once: Parakeet ≈ 490 MB (from the sherpa-onnx GitHub release, cached in
`~/.cache/streamlit-stt`, override with `STT_MODEL_DIR`), Whisper from Hugging Face (turbo ≈ 1.5 GB).
Live mic needs `localhost` or HTTPS.

## How it works
* `stt/base.py`: the `STTEngine.transcribe(audio) → Result` contract every model follows.
* `stt/parakeet_engine.py`: downloads/loads Parakeet; long audio is cut into ≤15 s pieces at the quietest point (Parakeet degrades on long, mixed-language chunks).
* `stt/whisper_engine.py`: faster-whisper wrapper.
* `stt/registry.py`: the model list shown in the sidebar. **Add a model = write an engine + add one line here.**
* `stt/live.py`: live streaming (resample → VAD → worker thread → partial/final events). Works with any engine.
* `app.py`: the Streamlit UI.

## Known limits
* Parakeet v3 is European-languages only. For Hindi or other Indian languages use Whisper (or add Nemotron 3.5, which lists Hindi).
* Parakeet is not a streaming model. Live mode re-transcribes the current sentence every ~1.5 s (partials) and once more on the pause (final).
* If a single recording switches between several languages, accuracy can drop. Keep chunks short (default 15 s).
* `.srt` cues from Parakeet are chunk-level (up to 15 s), not word-level.
* `av` is pinned `<17` because newer versions break faster-whisper's decoding.

## Put it on GitHub
```bash
git init && git add . && git commit -m "Streamlit CPU speech-to-text"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```
Host: Streamlit Community Cloud (main file `app.py`; memory is limited, Parakeet at ~1 GB RAM is the better fit there than Whisper turbo)
or any CPU VM / Hugging Face Space over HTTPS. Some networks need a TURN server for the Live tab (`rtc_configuration` in `app.py`).

## Tests
```bash
pip install pytest && python -m pytest -q                       # fast tests, no model needed
STT_RUN_MODEL_TESTS=1 python -m pytest -q tests/test_parakeet.py   # real model (downloads ~490 MB)
```
