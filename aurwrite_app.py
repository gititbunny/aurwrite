import streamlit as st
import whisper
import os
from datetime import datetime

# Folder setup
UPLOAD_FOLDER = "uploads"
TRANSCRIPT_FOLDER = "outputs/transcripts"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPT_FOLDER, exist_ok=True)

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

# UI Title
st.title("🎙️ Aurwrite - Audio to Story Remix")

# File uploader
uploaded_file = st.file_uploader("Upload a voice note (MP3 or WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}_{uploaded_file.name}")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(file_path, format="audio/wav")

    # Transcribe using Whisper
    st.info("Transcribing...")
    result = model.transcribe(file_path)
    transcript = result["text"]

    # Save transcript
    transcript_path = os.path.join(TRANSCRIPT_FOLDER, f"{timestamp}_transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # Display transcript
    st.success("Transcription complete!")
    st.text_area("📝 Transcript", transcript, height=250)
    
    from transformers import pipeline

# Load story style prompts
STYLE_PATHS = {
    "Fairy Tale": "styles/fairytale.txt",
    "News Article": "styles/news.txt",
    "Stand-Up Comedy": "styles/comedy.txt",
    "Horror": "styles/horror.txt"
}

# Select storytelling style
style = st.selectbox("✨ Choose a storytelling style", list(STYLE_PATHS.keys()))

# Button to rewrite
if st.button("🌀 Remix My Story"):
    # Load the prompt for the selected style
    with open(STYLE_PATHS[style], "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    # Combine prompt + transcript
    combined_input = f"{prompt}\n{transcript}"

    # Load local transformer model
    @st.cache_resource
    def load_rewrite_model():
        return pipeline("text2text-generation", model="google/flan-t5-small")

    rewrite_model = load_rewrite_model()

    # Generate styled text
    with st.spinner("Rewriting your story..."):
        output = rewrite_model(combined_input, max_length=500)[0]["generated_text"]

    # Show result
    st.success(f"Story rewritten as: {style}")
    st.text_area("🎭 Styled Story", output, height=300)

