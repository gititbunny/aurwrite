# Aurwrite — Audio to Story Remix

Local, zero‑cost pipeline:
- Whisper transcription (CPU)
- Style rewrite (GPT‑2 via `transformers`) — Fairy Tale / News / Comedy / Horror
- Offline TTS narration (`pyttsx3`)
- Streamlit UI (dark fantasy theme)

## Run
1. Install FFmpeg and add `ffmpeg/bin` to PATH (Windows).
2. `python -m pip install -r requirements.txt`
3. `streamlit run aurwrite_app.py`

## Folders
- `uploads/` user audio
- `outputs/transcripts/` raw text
- `outputs/audio/` generated narration
