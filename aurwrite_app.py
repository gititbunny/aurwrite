import os
import io
import base64
from datetime import datetime 
import streamlit as st
import whisper
from transformers.pipelines import pipeline
import pyttsx3

# Paths
BASE_DIR = os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "outputs", "transcripts")
AUDIO_OUT_DIR = os.path.join(BASE_DIR, "outputs", "audio")
STYLES_DIR = os.path.join(BASE_DIR, "styles")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

for p in [UPLOAD_DIR, TRANSCRIPT_DIR, AUDIO_OUT_DIR]:
    os.makedirs(p, exist_ok=True)

# Page config 
st.set_page_config(
    page_title="Aurwrite: Audio to Story Creator",
    page_icon=os.path.join(ASSETS_DIR, "icon.png"),
    layout="wide"
)

# Dependency check
import shutil
if shutil.which("ffmpeg") is None:
    st.error("FFmpeg not found. Please ensure it's installed.")
    st.stop()

# Warn if no TTS engine is available
if not (shutil.which("espeak-ng") or shutil.which("espeak")):
    st.warning("Text-to-speech may fail: espeak-ng not found on this system.")


# Custom styles 
HAND_FONT = "Shadows Into Light"
st.markdown(
    f"""
    <link href="https://fonts.googleapis.com/css2?family={HAND_FONT.replace(' ', '+')}:wght@400;700&display=swap" rel="stylesheet">
    <style>
      .aur-title {{ font-family: '{HAND_FONT}', cursive; font-size: 3rem; letter-spacing: .5px; }}
      .aur-subtle {{ opacity:.85 }}
      .stApp {{ background: radial-gradient(1200px 800px at 20% 0%, #1C3AFF18, transparent 60%), 
                          radial-gradient(1200px 800px at 100% 100%, #B131FA18, transparent 60%); }}
      .pill {{ display:inline-block; padding:.3rem .6rem; border-radius:999px; background:#4A0074; color:#fff; font-size:.8rem; }}
      .glass {{ background:rgba(10,7,18,.6); border:1px solid rgba(255,255,255,.06); border-radius:16px; padding:1rem; }}
      .footer a{{ color:#B131FA; text-decoration:none; }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar (tabbed feel)
with st.sidebar:
    st.image(os.path.join(ASSETS_DIR, "logo.png"), use_container_width=True)
    st.markdown("### Navigation")
    active_view = st.radio(
        "Go to",
        ["Create", "How it works", "About"],
        captions=[
            "Let the magic happen...",
            "More Information",
            "Developer Information"
        ]
    )
    st.markdown("---")
    st.markdown(
        "<span class='pill'>Dark Fantasy</span> <span class='pill'>Offline TTS</span> <span class='pill'>Local Whisper</span>",
        unsafe_allow_html=True
    )

# Cached models
@st.cache_resource(show_spinner=True)
def load_whisper():
    # small/base are okay on CPU; tiny is fastest
    return whisper.load_model("base")

@st.cache_resource(show_spinner=True)
def load_rewriter():
    # Use a lightweight, no-SentencePiece model.
    # GPT-2 isn't an instruction model, but works for fun style prompts.
    return pipeline("text-generation", model="gpt2", device=-1)

whisper_model = load_whisper()
rewrite_pipe = load_rewriter()

# Helpers
def save_bytes(file_bytes: bytes, folder: str, filename: str) -> str:
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path

import subprocess, shutil, tempfile

def tts_to_bytes(text: str) -> bytes:
    """Generate WAV bytes. Prefer espeak-ng on Linux; fallback to pyttsx3."""
    tmp_wav = os.path.join(AUDIO_OUT_DIR, "tmp_tts.wav")
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)

    # 1) Use espeak-ng if available (Linux/Streamlit Cloud)
    if shutil.which("espeak-ng"):
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            tf.write(text)
            tmp_txt = tf.name
        try:
            subprocess.run(
                ["espeak-ng", "-v", "en-us", "-s", "150", "-w", tmp_wav, "-f", tmp_txt],
                check=True,
            )
            with open(tmp_wav, "rb") as f:
                data = f.read()
            return data
        finally:
            try:
                os.remove(tmp_txt)
            except Exception:
                pass
            try:
                os.remove(tmp_wav)
            except Exception:
                pass

    # 2) Fallback to pyttsx3 (works on Windows/macOS)
    import pyttsx3
    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    engine.setProperty("rate", max(120, rate - 40))
    engine.save_to_file(text, tmp_wav)
    engine.runAndWait()

    if not os.path.exists(tmp_wav):
        raise RuntimeError("TTS failed. On Linux, install and use espeak-ng.")

    with open(tmp_wav, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp_wav)
    except Exception:
        pass
    return data


def dl_button(label: str, data: bytes, file_name: str, mime: str):
    b64 = base64.b64encode(data).decode()
    href = f'<a download="{file_name}" href="data:{mime};base64,{b64}" class="pill">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)

STYLE_FILES = {
    "Fairy Tale": os.path.join(STYLES_DIR, "fairytale.txt"),
    "News Article": os.path.join(STYLES_DIR, "news.txt"),
    "Stand-Up Comedy": os.path.join(STYLES_DIR, "comedy.txt"),
    "Horror": os.path.join(STYLES_DIR, "horror.txt"),
}

def load_style_prompt(style_name: str) -> str:
    path = STYLE_FILES[style_name]
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

# Views
if active_view == "Create":
    col_logo, col_title = st.columns([1, 4], vertical_alignment="center")
    with col_logo:
        st.image(os.path.join(ASSETS_DIR, "logo.png"), width=80)
    with col_title:
        st.markdown("<div class='aur-title'>Aurwrite: Audio to Story Creator</div>", unsafe_allow_html=True)
        st.caption("Upload a voice note → Transcribe with Whisper → Rewrite in a style → Hear it narrated")

    # Upload
    with st.container(border=True):
        st.subheader("🎧 Upload Audio")
        uploaded = st.file_uploader("MP3 or WAV (200MB or less)", type=["mp3", "wav"])
        style = st.selectbox("Choose a storytelling style", list(STYLE_FILES.keys()))

        if uploaded:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join([c for c in uploaded.name if c.isalnum() or c in (" ", ".", "_", "-")]).strip()
            audio_path = save_bytes(uploaded.read(), UPLOAD_DIR, f"{ts}_{safe_name}")
            st.audio(audio_path)

            if st.button("📝 Create Story", type="primary"):
                with st.status("Transcribing with Whisper…", expanded=True):
                    result = whisper_model.transcribe(audio_path)
                    transcript = result.get("text", "").strip()
                    st.write("Transcript length:", len(transcript))

                if not transcript:
                    st.error("Transcription came back empty. Try a clearer recording.")
                    st.stop()

                # Save transcript
                transcript_file = os.path.join(TRANSCRIPT_DIR, f"{ts}_transcript.txt")
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(transcript)

                # Style rewrite
                with st.status(f"Rewriting as **{style}**…", expanded=True):
                    prompt = load_style_prompt(style)
                    # Build a generator prompt for GPT‑2
                    seed_prompt = f"{prompt}\n\nOriginal:\n{transcript}\n\nRewrite:\n"
                    out = rewrite_pipe(
                        seed_prompt,
                        max_new_tokens=220,
                        temperature=0.9,
                        top_p=0.95,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=50256,
                    )[0]["generated_text"]
                    styled = out.split("Rewrite:\n", 1)[-1].strip()
                    st.write("Styled length:", len(styled))

                # Display
                left, right = st.columns(2)
                with left:
                    st.subheader("📝 Transcript")
                    st.text_area("Raw transcript", transcript, height=300)
                with right:
                    st.subheader(f"🎭 {style}")
                    st.text_area("Styled story", styled, height=300)

                # TTS
                with st.spinner("Generating narration…"):
                    wav_bytes = tts_to_bytes(styled)
                st.audio(wav_bytes, format="audio/wav")
                dl_button("⬇️ Download narration (WAV)", wav_bytes, f"{ts}_{style.lower().replace(' ','_')}.wav", "audio/wav")

                st.success("Done! You can tweak the style or upload another audio.")

elif active_view == "How it works":
    st.subheader("Pipeline")
    st.markdown(
        """
        Aurwrite is a fantasy-themed, dark-mode storytelling app built with Streamlit. Users upload a short audio file (like a voice note), which is transcribed with Whisper, rewritten in one of four styles: Fairy Tale, News Article, Stand-Up Comedy, or Horror, using a local LLM, and then narrated back using free TTS libraries (pyttsx3 or edge-tts). Features include a handwritten font style, fantasy-inspired dark UI, sidebar navigation, and live playback.
        """
    )

else:  # About
    st.subheader("About Aurwrite")
    st.markdown(
        """
        An AI-powered 'Audio-to-Story' creator tool that transforms voice notes into narrated stories in multiple styles.
        Built with ❤️ by Nina Nkhwashu.
        """
    )

# Footer
from datetime import datetime
year = datetime.now().year

st.markdown(
    f"""
    <style>
      .weathif-footer {{
        text-align: center;
        margin-top: 24px;
        padding: 12px 0;
        opacity: .8;
        font-size: 14px;
        color: white;
      }}
      .weathif-footer a {{
        color: #1C3AFF;
        text-decoration: none;
      }}
    </style>
    <div class="weathif-footer">
      © {year} All rights reserved. Built by
      <a href="https://www.linkedin.com/in/ninankhwashu/" target="_blank" rel="noopener">Nina Nkhwashu</a>.
    </div>
    """,
    unsafe_allow_html=True
)

