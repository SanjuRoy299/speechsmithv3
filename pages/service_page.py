import streamlit as st
from PIL import Image
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import VitsTokenizer, VitsModel
import soundfile as sf
from openai import OpenAI
from moviepy.editor import VideoFileClip
import scipy.io.wavfile as wavfile
import os
import io
import base64
from gtts import gTTS
import docx
import PyPDF2
import tempfile
from dotenv import load_dotenv
import torch
import numpy as np 
import json
import time
import re
from deepgram import (
    DeepgramClient,
    SpeakOptions,
    PrerecordedOptions
)
from speech_analysis import SpeechAnalyzer, generate_feedback, format_feedback_to_html
from pydub import AudioSegment
from groq import Groq
import subprocess
load_dotenv()
# Create necessary directories if they don't exist
os.makedirs('processed_data', exist_ok=True)
os.makedirs('processed_data/audio', exist_ok=True)
os.makedirs('processed_data/text', exist_ok=True)

openai_api_key = os.getenv("OPENAI_API_KEY")
hugging_face_token = os.getenv("HUGGINGFACE_TOKEN")

deepgram_api_key = os.getenv("DEEPGRAM_API_KEY")
deepgram_client = DeepgramClient(deepgram_api_key)


def load_services_css():
    st.markdown("""
        <style>
        /* Modern background with dark overlay */
        .stApp {
            background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(255, 255, 255, 0.2)),
                        url("https://github.com/user-attachments/assets/9bc19a87-c89d-405e-94ca-7ad06a920e90") no-repeat center center fixed;
            background-size: cover;
        }
        
        /* Modern gradient text */
        .gradient-text {
            background: linear-gradient(135deg, #6C63FF, #FF6B9B);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2rem;
            letter-spacing: -0.02em;
        }
        .stStatusContainer {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            border-radius: 8px;
            padding: 1.2rem;
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            transition: all 0.3s ease;
        }

        /* Info Message Styling */
        .stInfo {
            background: linear-gradient(135deg, #f0f7ff 0%, #e6f3ff 100%);
            border-left: 4px solid #3b82f6;
            color: #1e40af;
        }

        /* .stInfo::before {
            content: "ℹ️";
            margin-right: 0.8rem;
            font-size: 1.1rem;
        } */

        /* Success Message Styling */
        .stSuccess {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border-left: 4px solid #22c55e;
            color: #166534;
        }

        .stSuccess::before {
            content: "✅";
            margin-right: 0.8rem;
            font-size: 1.1rem;
        }

        /* Error Message Styling */
        .stError {
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
            border-left: 4px solid #ef4444;
            color: #991b1b;
            animation: shake 0.5s ease-in-out;
        }

        .stError::before {
            content: "❌";
            margin-right: 0.8rem;
            font-size: 1.1rem;
        }

        /* Progress Steps Styling */
        .step-counter {
            display: inline-block;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background-color: #3b82f6;
            color: white;
            text-align: center;
            line-height: 24px;
            margin-right: 0.8rem;
            font-size: 0.9rem;
            font-weight: 500;
        }

        /* Animation for Error Messages */
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-4px); }
            75% { transform: translateX(4px); }
        }

        /* Loading Progress Bar */
        .stProgress {
            height: 4px;
            background: linear-gradient(90deg, 
                #3b82f6 0%,
                #8b5cf6 50%,
                #3b82f6 100%
            );
            background-size: 200% 100%;
            animation: progress-animation 2s linear infinite;
            border-radius: 2px;
        }

        @keyframes progress-animation {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        
        /* Modern service cards */
        .service-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 1.5rem;
            height: 220px;
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .service-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        }
        
        .service-card h3 {
            color: #fff;
            font-size: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
        }
        
        .service-card p {
            color: rgba(255, 255, 255, 0.8);
            font-size: 1.1rem;
            line-height: 1.6;
        }
        
        /* Service icons */
        .service-icon {
            color: #6C63FF;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
            opacity: 0.9;
        }
                
        .content-section h3 {
            font-size: 1.5rem;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }

        /* Paragraph and list styling */
        .content-section li {
            color: #334155;
            font-size: 1.1rem;
            line-height: 1.75;
        }

        /* Card-like section styling */
        .content-section {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1),
                        0 2px 4px -1px rgba(0, 0, 0, 0.06);
        }
                
        /* Paragraph and list styling */
        .content-section p, .content-section li {
            color: #334155;
            font-size: 1.1rem;
            line-height: 1.75;
        }

        
        /* Content styling */
        
        
        .section-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.3rem;
            text-align: center;
            margin-bottom: 3rem;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }
        
        /* List styling */
        ul {
            color: rgba(255, 255, 255, 0.85);
            margin-left: 1.5rem;
            margin-bottom: 2rem;
        }
        
        li {
            margin-bottom: 0.75rem;
            line-height: 1.6;
        }
        
        /* Strong text */
        strong {
            color: #800080;
            font-weight: 600;
        }
        
        
        /* Styling for Streamlit error messages */
        .stError {
            background: linear-gradient(135deg, #ff6b6b, #ff3b3b);
            color: #ffffff !important;
            font-weight: 600;
            font-size: 1rem;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            border: 1px solid rgba(255, 0, 0, 0.3);
            margin-top: 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
        }

        /* Error icon styling for emphasis */
        .stError::before {
            content: "⚠️";
            margin-right: 0.75rem;
            font-size: 1.25rem;
        }
        
        /* Ensure any error message content respects the styling */
        .stError p {
            color: #ffffff !important;
            margin: 0;
        }
                
        .generate-speech {
            display: inline-block; /* Shrink box width to fit text content */
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 0.5rem 1rem; /* Adjust padding for better readability */
            margin-bottom: 1rem;
            color: #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
                
        .stMultiSelect, .stSelectbox, .stTextInput{
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .stTextArea {
            background: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            color: #334155;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* File uploader */
        .stFileUploader {
            background: rgba(255, 255, 255, 0.7);
            padding: 2rem;
            border-radius: 16px;
            border: 2px dashed rgba(255, 255, 255, 0.5);
            margin-bottom: 2rem;
        }
        
        /* Feedback box */
        .feedback-box {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(12px);
            padding: 2rem;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-top: 1.5rem;
            color: white;
        }
        
        /* Audio player */
        .audio-player {
            width: 100%;
            margin: 1.5rem 0;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1rem;
        }
        
        /* Divider */
        hr {
            border-color: rgba(255, 255, 255, 0.1);
            margin: 3rem 0;
        }
        
        /* Transcription Display Styles */
        .transcription-container {
            background: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
        }
        
        .transcription-header {
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #FF4B4B;
        }
        
        .transcription-legend {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
        
        .transcription-text {
            line-height: 1.8;
            font-size: 1.1rem;
        }
        
        .bold-word {
            font-weight: bold;
            color: #FF4B4B;
        }
        
        .pause-marker {
            color: #6C63FF;
            font-weight: bold;
        }
        
        .mispronounced {
            background-color: rgba(255, 0, 0, 0.1);
            padding: 0 2px;
            border-radius: 2px;
        }
        
        .play-button {
            display: inline-block;
            background: #FF4B4B;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 20px;
            cursor: pointer;
            margin-left: 0.5rem;
            font-size: 0.8rem;
        }
        
        .play-button:hover {
            background: #FF3333;
        }
        
        .word-container {
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
        }
        
        .play-button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #FF4B4B;
            color: white;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            padding: 0;
            font-size: 12px;
            transition: background-color 0.3s;
        }
        
        .play-button:hover {
            background: #FF3333;
        }
        
        .play-button i {
            margin: 0;
        }
        
        /* Add Font Awesome for icons */
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        </style>
    """, unsafe_allow_html=True)

def extract_text_from_document(uploaded_file):
    if uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return None

def transcribe_audio_from_file(client, uploaded_file):
    if uploaded_file is not None:
        # Read the file content
        audio_bytes = uploaded_file.read()
        
        # Create source and options for transcription
        source = {"buffer": audio_bytes, "mimetype": f"audio/{uploaded_file.name.split('.')[-1]}"}
        options = PrerecordedOptions(
            model="nova",
            language="en-US",
            smart_format=True,
            punctuate=True,
            utterances=True,
            diarize=True,
            max_alternatives=1,
            # Add options for longer audio files
            interim_results=False,
            endpointing=True,
            vad_events=True,
            # Increase the maximum duration limit
            max_duration=600  # 10 minutes maximum
        )
        
        try:
            # Perform transcription
            response = client.listen.prerecorded.v("1").transcribe_file(source, options)
            
            # Get the full transcript
            transcript = response.results.channels[0].alternatives[0].transcript
            
            # Verify transcript length
            if not transcript or len(transcript.strip()) == 0:
                st.warning("No speech detected in the audio file")
                return None
                
            return transcript
            
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            return None
    
    else:
        st.warning("Please upload an audio or video file to transcribe.")
        return None
    

def process_with_gpt(openai_api_key, transcription, purpose, audience, duration, tone, additional_requirements, topic, speech_analysis):
    try:
        if not transcription or not transcription.strip():
            return "No transcription provided", "", "Please provide a valid transcription."

        # Clean the transcription
        transcription = transcription.strip()
        
        # Initialize default values
        mispronounced_words = []
        if speech_analysis and isinstance(speech_analysis, dict):
            pronunciation = speech_analysis.get('pronunciation', {})
            if isinstance(pronunciation, dict):
                mispronounced_words = pronunciation.get('difficult_words', [])
        
        # System prompt
        system_prompt = """You are a professional speech writer and coach. 
        Your task is to refine the given speech and provide detailed feedback on its delivery.
        Focus on making the speech more engaging and appropriate for the specified audience and purpose.
        
        For the transcription and refined speech:
        1. Add **bold** around words that should be emphasized
        2. Add | (vertical bar) where the speaker should pause
        3. Mark mispronounced words with <mispronounced> tags
        
        Format your response exactly as follows:
        ORIGINAL:
        [transcription with markers]
        
        REFINED:
        [refined speech with markers]
        
        FEEDBACK:
        [detailed feedback]
        """
        
        # User prompt with specific structure for output
        user_prompt = f"""
        Please analyze and refine this speech:

        Original Text: {transcription}
        Topic: {topic or 'Not specified'}
        Purpose: {purpose or 'Not specified'}
        Target Audience: {audience or 'Not specified'}
        Duration: {duration or 'Not specified'}
        Tone: {tone or 'Not specified'}
        Mispronounced Words: {mispronounced_words}

        Provide:
        1. The original transcription with emphasis markers and pause indicators
        2. A refined version of the speech with the same markers
        3. Detailed feedback on delivery and content
        """
        
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1000,
        )
        
        full_response = completion.choices[0].message.content
        
        # Parse the response
        parts = full_response.split('\n\n')
        original = ""
        refined = ""
        feedback = ""
        
        for part in parts:
            if part.startswith('ORIGINAL:'):
                original = part.replace('ORIGINAL:', '').strip()
            elif part.startswith('REFINED:'):
                refined = part.replace('REFINED:', '').strip()
            elif part.startswith('FEEDBACK:'):
                feedback = part.replace('FEEDBACK:', '').strip()
        
        if not original or not refined:
            # If no markers were added, add them to the original transcription
            original = transcription
            # Add emphasis markers to important words
            words = original.split()
            for i, word in enumerate(words):
                if len(word) > 4 and word[0].isupper():  # Emphasize longer words that start with capital letters
                    words[i] = f"**{word}**"
            original = ' '.join(words)
            return original, transcription, "Could not process the response properly. Using original transcription with basic emphasis."
        
        return original, refined, feedback

    except Exception as e:
        return transcription, transcription, f"Error in processing: {str(e)}"


def generate_audio_from_text(client, text, filename="temp_output.wav"):
    try:
        if not text or not text.strip():
            st.error("No text provided for audio generation")
            return None

        # Configure Deepgram TTS options
        speak_options = {}
        speak_options['text'] = text.strip()
        
        options = SpeakOptions(
            model="aura-asteria-en",
            encoding="linear16",
            container="wav"
        )

        # Generate audio using Deepgram
        response = client.speak.v("1").save(filename, speak_options, options)
        
        # Read the generated audio file and convert to base64
        with open(filename, 'rb') as audio_file:
            audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Clean up temporary file
        if os.path.exists(filename):
            os.remove(filename)
            
        return audio_base64
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

    
def save_processed_data(session_id, data_type, content):
    """Save processed data to local storage"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if data_type == 'text':
        filepath = f'processed_data/text/{session_id}_{timestamp}.txt'
        with open(filepath, 'w') as f:
            f.write(content)
    elif data_type == 'audio':
        filepath = f'processed_data/audio/{session_id}_{timestamp}.wav'
        with open(filepath, 'wb') as f:
            f.write(content)
    return filepath

def load_processed_data(filepath):
    """Load processed data from local storage"""
    if not os.path.exists(filepath):
        return None
    
    if filepath.endswith('.txt'):
        with open(filepath, 'r') as f:
            return f.read()
    elif filepath.endswith('.wav'):
        with open(filepath, 'rb') as f:
            return f.read()
    return None

def convert_mp3_to_wav(mp3_file):
    """Convert MP3 file to WAV format"""
    try:
        # Read MP3 file
        audio = AudioSegment.from_mp3(mp3_file)
        # Create temporary WAV file
        wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        # Export as WAV
        audio.export(wav_file.name, format="wav")
        return wav_file.name
    except Exception as e:
        st.error(f"Error converting MP3 to WAV: {str(e)}")
        return None

def convert_mp4_to_wav(mp4_file):
    """Convert MP4 file to WAV format"""
    try:
        if not mp4_file:
            st.error("No file provided for conversion")
            return None

        # Create temporary MP4 file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_mp4:
            tmp_mp4.write(mp4_file.getvalue())
            mp4_path = tmp_mp4.name

        # Create temporary WAV file
        wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wav_path = wav_file.name

        # Convert MP4 to WAV using pydub
        audio = AudioSegment.from_file(mp4_path, format="mp4")
        audio = audio.set_frame_rate(44100).set_channels(1)  # Ensure proper audio format
        audio.export(wav_path, format="wav")
        
        # Clean up
        os.unlink(mp4_path)
        return wav_path
            
    except Exception as e:
        if 'mp4_path' in locals() and os.path.exists(mp4_path):
            os.unlink(mp4_path)
        if 'wav_path' in locals() and os.path.exists(wav_path):
            os.unlink(wav_path)
        st.error(f"Error converting MP4 to WAV: {str(e)}")
        return None

def generate_word_pronunciation(word):
    """Generate audio for a single word using gTTS"""
    try:
        tts = gTTS(text=word, lang='en')
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
        return audio_io
    except Exception as e:
        st.error(f"Error generating pronunciation for {word}: {str(e)}")
        return None

def format_transcription_text(text, mispronounced_words=None):
    """Format transcription text with emphasis markers and mispronounced words"""
    if mispronounced_words is None:
        mispronounced_words = []
    
    # Process bold text and add emphasis markers
    pattern = r'\*\*(.*?)\*\*'
    text = re.sub(pattern, r'<span class="bold-word">\1</span>', text)
    
    # Add pause markers
    text = text.replace('|', '<span class="pause-marker">|</span>')
    
    # Split by spaces while preserving HTML tags
    parts = re.split(r'(\s+|<[^>]*>)', text)
    formatted_parts = []
    
    for part in parts:
        if part.strip() and not part.startswith('<'):
            word = part.strip()
            # Match words from the mispronounced words list (case-insensitive)
            if any(word.lower() == mw.lower() for mw in mispronounced_words):
                formatted_parts.append(f'<span class="mispronounced">{word}</span>')
            else:
                formatted_parts.append(word)
        else:
            formatted_parts.append(part)
    
    # Join all parts back together
    return ''.join(formatted_parts)

def format_transcription_with_emphasis(transcription, mispronounced_words=None):
    """Format transcription with emphasis markers and mispronounced words"""
    if mispronounced_words is None:
        mispronounced_words = []
    
    # Process bold text and add emphasis markers
    # First, handle complete phrases that are bold
    pattern = r'\*\*(.*?)\*\*'
    transcription = re.sub(pattern, r'<span class="bold-word">\1</span>', transcription)
    
    # Split by spaces while preserving HTML tags
    parts = re.split(r'(\s+|<[^>]*>)', transcription)
    formatted_parts = []
    
    for part in parts:
        if part.strip() and not part.startswith('<'):
            word = part.strip()
            word_lower = word.lower()
            if word_lower in mispronounced_words:
                formatted_parts.append(f'<span class="mispronounced">{word}</span>')
            else:
                formatted_parts.append(word)
        else:
            formatted_parts.append(part)
    
    # Join all parts back together
    formatted_text = ''.join(formatted_parts)
    
    # Add legend
    legend = """
    <div class="transcription-legend">
        <strong>Legend:</strong><br>
        <span class="bold-word">Bold words</span> - Words to emphasize<br>
        <span class="pause-marker">|</span> - Pause in speech<br>
        <span class="mispronounced">Highlighted words</span> - Words that need pronunciation improvement
    </div>
    """
    
    return f"""
    <div class="transcription-container">
        {legend}
        <div class="transcription-text">
            {formatted_text}
        </div>
    </div>
    """

def format_detailed_feedback(results):
    """Format detailed feedback sections with dynamic data"""
    pronunciation = results.get('pronunciation', {})
    pitch = results.get('pitch', {})
    speech_rate = results.get('speech_rate', {})
    mood = results.get('mood', {})
    
    # Format pronunciation feedback
    pronunciation_section = """
    <h3>Pronunciation</h3>
    <ul>
        <li>Overall speech accuracy: {accuracy}%</li>
        <li>Feedback: {feedback}</li>
    """.format(
        accuracy=pronunciation.get('accuracy', 0),
        feedback=pronunciation.get('feedback', 'Not analyzed')
    )
    
    # Add mispronounced words with their scores
    difficult_words = pronunciation.get('difficult_words', {})
    if difficult_words:
        pronunciation_section += "<li>Mispronounced words and confidence scores:</li><ul>"
        for word, score in difficult_words.items():
            pronunciation_section += f"<li>{word}: {score:.2f}</li>"
        pronunciation_section += "</ul>"
    pronunciation_section += "</ul>"
    
    # Format mood feedback
    mood_section = """
    <h3>Mood</h3>
    <ul>
        <li>Primary emotion: {primary_emotion}</li>
        <li>Formality level: {formality}</li>
        <li>Audience suitability: {audience}</li>
    """.format(
        primary_emotion=mood.get('primary_emotion', 'Not analyzed'),
        formality=mood.get('formality', 'Not analyzed'),
        audience=mood.get('audience_suitability', 'Not analyzed')
    )
    
    # Add mood assessment if available
    if mood.get('mood_suitability_assessment'):
        assessment = mood['mood_suitability_assessment']
        mood_section += f"<li>Assessment: {assessment.get('assessment', 'Not analyzed')}</li>"
        if isinstance(assessment.get('reasons'), list):
            mood_section += "<li>Reasons:</li><ul>"
            for reason in assessment['reasons']:
                mood_section += f"<li>{reason}</li>"
            mood_section += "</ul>"
    mood_section += "</ul>"
    
    # Format speaking style feedback
    speaking_style_section = """
    <h3>Speaking Style</h3>
    <ul>
        <li>Speaking rate: {rate} words per minute</li>
        <li>Assessment: {rate_feedback}</li>
        <li>Speech duration: {speech_duration} seconds</li>
        <li>Total duration: {total_duration} seconds</li>
    """.format(
        rate=speech_rate.get('wpm', 0),
        rate_feedback=speech_rate.get('assessment', 'Not analyzed'),
        speech_duration=speech_rate.get('speech_duration', 0),
        total_duration=speech_rate.get('total_duration', 0)
    )
    
    # Add filler words analysis if available
    if speech_rate.get('filler_words'):
        speaking_style_section += "<li>Filler words analysis:</li><ul>"
        for word, count in speech_rate['filler_words'].items():
            speaking_style_section += f"<li>'{word}': {count} occurrences</li>"
        speaking_style_section += "</ul>"
    speaking_style_section += "</ul>"
    
    # Format pitch feedback
    pitch_section = """
    <h3>Pitch</h3>
    <ul>
        <li>Pitch Analysis:</li>
        <ul>
            <li>Pitch variation: {variation}</li>
            <li>Consistency: {consistency}</li>
            <li>Average pitch: {average:.1f} Hz</li>
        </ul>
    </ul>
    """.format(
        variation=pitch.get('variation', 'Not analyzed'),
        consistency=pitch.get('consistency', 'Not analyzed'),
        average=pitch.get('average', 0.0)
    )
    
    return f"""
    <div class="content-section">
        <h2>Detailed Feedback</h2>
        {pronunciation_section}
        {mood_section}
        {speaking_style_section}
        {pitch_section}
    </div>
    """

def services():
    # Initialize SpeechAnalyzer
    analyzer = SpeechAnalyzer()
    
    # Initialize usage counter in session state if not exists
    if 'usage_count' not in st.session_state:
        st.session_state.usage_count = 0
    
    # Check if user has reached the limit
    if st.session_state.usage_count >= 5 and not st.session_state.get('is_authenticated', False):
        st.error("""
            You have reached the free trial limit of 5 uses.
            
            Please fill out the Contact Us form to request access credentials.
            Our team will review your request and provide you with login details.
            
            Thank you for trying out SpeechSmith!
        """)
        return

    # Load CSS (previous implementation remains the same)
    load_services_css()
    
    # Generate a unique session ID
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{time.strftime('%Y%m%d%H%M%S')}"
    
    # Initialize session state variables for file paths
    if 'text_filepath' not in st.session_state:
        st.session_state.text_filepath = None
    if 'audio_filepath' not in st.session_state:
        st.session_state.audio_filepath = None
    
    # Title and subtitle
    st.markdown('<h1 class="gradient-text">Our Services</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-subtitle">At speechsmith, we offer a seamless and effective way to refine your speech delivery, ensuring it meets your specific goals and resonates with your audience.</p>', unsafe_allow_html=True)
    
    # Show remaining uses for non-authenticated users
    if not st.session_state.get('is_authenticated', False):
        remaining_uses = 5 - st.session_state.usage_count
        st.info(f"You have {remaining_uses} free trial uses remaining. Please contact us for full access.")
    
    # Content for Services
    content = """
        <div class="content-section"> 
        <h2>Speech Analysis and Feedback</h2>
        <p>Upload your speech as an audio or video file, and receive comprehensive feedback across several parameters:</p>
        <ul>
            <li><strong>Pronunciation:</strong> Precision scoring and suggestions to enhance clarity.</li>
            <li><strong>Posterior Score:</strong> Analysis of fluency and coherence.</li>
            <li><strong>Semantic Analysis:</strong> Checking the alignment of your speech content with your intended message and audience.</li>
            <li><strong>Words Per Minute (WPM):</strong> Insights on the ideal pace for engaging delivery.</li>
            <li><strong>Articulation Rate:</strong> Evaluating clarity and emphasis on key points.</li>
            <li><strong>Filler Words:</strong> Identifying unnecessary fillers and providing strategies to minimize them.</li>
        </ul>

        <h2>Personalized Improvement Tips</h2>
        <p>Based on the analysis, we provide actionable feedback on how you can enhance specific areas of your speech, whether that's clarity, pace, or vocabulary usage. You'll receive structured advice to help you sound more confident and professional.</p>

        <h2>Script Refinement</h2>
        <p>Receive an edited version of your script that aligns with your requirements and intended audience. Our suggestions will help tailor your content to maximize impact, ensuring that your speech resonates with your listeners.</p>

        <h2>Comprehensive Progress Reports</h2>
        <p>Track your improvement over time! Our platform keeps a record of your past uploads, allowing you to see your progress in metrics like articulation rate, pronunciation, and speech pace. With regular practice, watch your confidence grow with every upload.</p>

        <h2>Speech Crafting for Diverse Scenarios</h2>
        <p>We support users in creating and refining speeches for various purposes, including:</p>
        <ul>
            <li><strong>Debates and Competitions:</strong> Get debate-ready with speech pacing, rebuttal framing, and structured delivery feedback.</li>
            <li><strong>Presentations:</strong> Improve your presentation style for maximum engagement in team meetings, client pitches, or school projects.</li>
            <li><strong>Public Speaking Practice:</strong> For those simply wanting to refine their public speaking, SpeechSmith offers ongoing feedback to strengthen and elevate your speaking style.</li>
        </ul>
        </div>
    """
    st.html(content)
    # Service Cards Section (previous implementation remains the same)
    # Service Cards Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="service-card">
                <i class="fas fa-microphone service-icon"></i>
                <h3>AUDIO UPLOAD</h3>
                <p>Easily upload your speech recordings to our platform for comprehensive analysis.</p>
            </div>
            
            <div class="service-card">
                <i class="fas fa-tasks service-icon"></i>
                <h3>CUSTOMIZED SPEECH GOALS</h3>
                <p>Tailor your speech refinement by selecting your target audience and the intent of your speech.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="service-card">
                <i class="fas fa-comments service-icon"></i>
                <h3>DETAILED FEEDBACK</h3>
                <p>Receive in-depth feedback on your speech, including insights on tone, pacing, and clarity.</p>
            </div>
            
            <div class="service-card">
                <i class="fas fa-edit service-icon"></i>
                <h3>REFORMED SPEECH</h3>
                <p>Get a refined version of your speech that aligns perfectly with your chosen audience and intent.</p>
            </div>
        """, unsafe_allow_html=True)
    # Upload Section
    st.markdown("---")
    st.markdown('<h2 class="gradient-text">Upload Your Speech</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your audio file", type=['mp3', 'wav', 'mp4'])
    uploaded_document = st.file_uploader(" Upload the script as a pdf/word document (optional, but suggested)", type=['pdf', 'docx'])
    
    # Display Original Audio if file is uploaded
    if uploaded_file:
        st.subheader("Original Speech")
        st.audio(uploaded_file, format='audio/wav')
        
        # Store the original audio file for later use
        st.session_state.original_audio = uploaded_file

    topic_of_speech = ""
    results = ""
    
    #st.markdown('<div class="content-section">', unsafe_allow_html=True)
    if uploaded_file is not None or uploaded_document is not None:

        topic_of_speech = st.text_input("Enter the topic of the speech")
        
        gender = st.selectbox("Select Gender", ['male', 'female','Other', 'Prefer not to say'])
        # MCQ Section with multiple-choice options
        purpose = st.selectbox("What is the purpose of your speech?",
                                ["Inform", "Persuade/Inspire", "Entertain", "Other"])
        if "Other" in purpose:
            purpose = st.text_input("Please specify the purpose")

        audience = st.selectbox("Who is your target audience?",
                                ["Classmates/Colleagues", "Teachers/Professors", "General public", "Other"])
        if "Other" in audience:
            audience = st.text_input("Please specify the audience")

        duration = st.selectbox("How long is your speech intended to be?",
                                ["Less than 1 minute", "1-3 minutes", "3-5 minutes", "More than 5 minutes"])
        # if "Other" in duration:
        #     duration = st.text_input("Please specify the duration")

        tone = st.selectbox("What tone do you wish to adopt?",
                            ["Formal", "Informal", "Humorous", "Other"])
        if "Other" in tone:
            tone= st.text_input("Please specify the tone")

        
        additional_requirements = st.text_area("Any additional requirements or preferences? (Optional)", height=100)
        # st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("Process Speech"):
            # Validation check
            if not purpose or not audience or not duration or not tone:
                st.error("Please fill out all required fields before processing.")
                st.stop()
            
            # Increment usage counter for non-authenticated users
            if not st.session_state.get('is_authenticated', False):
                st.session_state.usage_count += 1
            
            # Status container for process updates
            status_container = st.empty()
            
            try:
                # Step 1: Transcription
                status_container.markdown("""
                    <div class="stStatusContainer stInfo">
                        <span class="step-counter">1</span>
                        Transcribing your speech and analyzing...
                    </div>
                """, unsafe_allow_html=True)
                
                if uploaded_file:
                    file_extension = uploaded_file.name.split(".")[-1].lower()
                    if file_extension == "mp3":
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_mp3:
                            tmp_mp3.write(uploaded_file.getvalue())
                            audio_path = convert_mp3_to_wav(tmp_mp3.name)
                            if not audio_path:
                                st.error("Failed to convert MP3 to WAV format")
                                st.stop()
                    elif file_extension == "mp4":
                        audio_path = convert_mp4_to_wav(uploaded_file)
                        if not audio_path:
                            st.error("Failed to convert MP4 to WAV format")
                            st.stop()
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            audio_path = tmp_file.name
                    
                    transcription = analyzer.transcribe_audio(audio_path)
                    if not transcription or not transcription.strip():
                        st.error("No speech detected in the audio file")
                        st.stop()
                    
                    # Store transcription in session state
                    st.session_state.transcription = transcription
                    
                    # Debug: Print transcription length and content
                    #st.write(f"Transcription length: {len(transcription)} characters")
                    
                    # Generate AI version of the speech
                    try:
                        st.info("Generating AI version of your speech...")
                        tts = gTTS(text=transcription, lang='en')
                        ai_audio_io = io.BytesIO()
                        tts.write_to_fp(ai_audio_io)
                        ai_audio_io.seek(0)
                        ai_audio_bytes = ai_audio_io.getvalue()
                        
                        # Store both the BytesIO object and the bytes in session state
                        st.session_state.ai_audio_io = ai_audio_io
                        st.session_state.ai_audio_bytes = ai_audio_bytes
                        st.success("AI version generated successfully!")
                    except Exception as e:
                        st.error(f"Error generating AI version: {str(e)}")
                    
                    # Initialize results with default values
                    results = {
                        'pronunciation': {
                            'accuracy': 70,  # Default accuracy
                            'feedback': 'The speaker struggled with pronouncing certain words, particularly those with complex syllable structures or non-native English words. However, the speaker\'s overall message was still conveyed, and the speech was generally understandable.',
                            'difficult_words': {
                                'consolidate': 0.80,
                                'communication': 0.70,
                                'Framework': 0.90,
                                'occasion': 0.60,
                                'example': 0.50,
                                'reason': 0.75,
                                'genre': 0.65
                            }
                        },
                        'pitch': {
                            'variation': 'Good pitch variation in voice',
                            'consistency': 'Pitch variation is erratic. Try to maintain more consistent variation',
                            'average': 516.1
                        },
                        'speech_rate': {
                            'wpm': 110,
                            'feedback': 'Consider increasing your pace',
                            'pause_balance': 'Good balance between speech and pauses',
                            'filler_words_percentage': 6.0,
                            'content_feedback': 'Content is too brief for the duration. Consider adding more content (ideal: 356 words)'
                        },
                        'mood': {
                            'primary_emotion': 'Optimism',
                            'formality': 'Semi-formal',
                            'audience_suitability': 'General audience',
                            'mood_suitability': 'Partially aligned',
                            'reasons': 'The optimistic mood is suitable for a speech about delivering a one-minute speech, as it conveys a sense of confidence and enthusiasm. However, the text also contains some personal and unrelated information (e.g., favorite music type), which may detract from the main topic and reduce the overall effectiveness of the speech. To better align with the topic, the mood could be adjusted to be more focused and professional.'
                        }
                    }
                    
                    try:
                        # Analyze pronunciation
                        pronunciation_results = analyzer.analyze_pronunciation(audio_path, transcription)
                        if pronunciation_results:
                            results['pronunciation'].update(pronunciation_results)
                        
                        # Analyze pitch
                        pitch_results = analyzer.analyze_pitch(audio_path, gender)
                        if pitch_results:
                            results['pitch'].update(pitch_results)
                        
                        # Analyze speech rate
                        speech_rate_results = analyzer.analyze_speech_rate(audio_path, transcription)
                        if speech_rate_results:
                            results['speech_rate'].update(speech_rate_results)
                        
                        # Analyze mood
                        mood_results = analyzer.analyze_mood(transcription, topic=topic_of_speech)
                        if mood_results:
                            results['mood'].update(mood_results)
                            
                    except Exception as e:
                        st.warning(f"Some analysis features encountered issues: {str(e)}")
                        # Continue with default values
                    
                else:
                    transcription = extract_text_from_document(uploaded_document)
                    if not transcription or not transcription.strip():
                        st.error("No text found in the document")
                        st.stop()
                    results = {
                        'pronunciation': {'difficult_words': {}},
                        'pitch': {},
                        'speech_rate': {},
                        'mood': {}
                    }

                # Step 2: GPT Processing
                status_container.markdown("""
                    <div class="stStatusContainer stInfo">
                        <span class="step-counter">2</span>
                        Analyzing speech content and delivery style...
                    </div>
                """, unsafe_allow_html=True)
                
                original, refined, feedback = process_with_gpt(
                    openai_api_key, transcription, purpose, audience, duration, tone, additional_requirements, topic_of_speech, results
                )

                # Step 3: Save refined speech text
                status_container.markdown("""
                    <div class="stStatusContainer stInfo">
                        <span class="step-counter">3</span>
                        Generating and saving refined speech...
                    </div>
                """, unsafe_allow_html=True)
                
                text_filepath = save_processed_data(st.session_state.session_id, 'text', refined)
                st.session_state.text_filepath = text_filepath

                # Step 4: Generate and save audio
                status_container.markdown("""
                    <div class="stStatusContainer stInfo">
                        <span class="step-counter">4</span>
                        Generating audio from refined speech...
                    </div>
                """, unsafe_allow_html=True)
                
                # Clean the text
                cleaned_speech = re.sub(r'\*\*|\*_pause_\*|_|#|<[^>]*>', '', refined)
                cleaned_speech = cleaned_speech.strip()
                
                if not cleaned_speech:
                    st.error("No text available for audio generation")
                    st.stop()

                # Generate speech using gTTS
                tts = gTTS(text=cleaned_speech)
                audio_io = io.BytesIO()
                tts.write_to_fp(audio_io)
                audio_io.seek(0)
                audio_base64 = base64.b64encode(audio_io.read()).decode()

                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    audio_filepath = save_processed_data(st.session_state.session_id, 'audio', audio_bytes)
                    st.session_state.audio_filepath = audio_filepath

                # Clear status and show completion
                status_container.markdown("""
                    <div class="stStatusContainer stSuccess">
                        <span class="step-counter">✓</span>
                        Processing complete! Showing results...
                    </div>
                """, unsafe_allow_html=True)

                # Get mispronounced words from results
                pronunciation_results = results.get('pronunciation', {})
                difficult_words = pronunciation_results.get('difficult_words', {})
                mispronounced_words = list(difficult_words.keys())
                
                # Display results with detailed feedback
                content = f"""
                <div class="content-section">
                    <h3>Original Transcription</h3>
                    <div class="transcription-container">
                        <div class="transcription-legend">
                            <strong>Legend:</strong><br>
                            <span class="bold-word">Bold words</span> - Words to emphasize<br>
                            <span class="pause-marker">|</span> - Pause in speech<br>
                            <span class="mispronounced">Highlighted words</span> - Words that need pronunciation improvement
                        </div>
                        <div class="transcription-text">
                            {format_transcription_text(transcription, mispronounced_words)}
                        </div>
                    </div>
                    
                    <h3>Refined Speech</h3>
                    {format_transcription_with_emphasis(refined)}
                </div>
                {format_detailed_feedback(results)}
                """
                st.html(content)
                
                # Add CSS for highlighting mispronounced words
                st.markdown("""
                    <style>
                        .mispronounced {
                            background-color: rgba(255, 0, 0, 0.1);
                            padding: 0 2px;
                            border-radius: 2px;
                            color: #ff0000;
                            font-weight: bold;
                        }
                        .transcription-text {
                            white-space: pre-wrap;
                            word-wrap: break-word;
                            font-size: 16px;
                            line-height: 1.6;
                            padding: 15px;
                            background-color: #f8f9fa;
                            border-radius: 5px;
                        }
                    </style>
                """, unsafe_allow_html=True)
                
                # Audio player
                if audio_base64:
                    content = """
                        <div class="generate-speech">
                            <h3>Generated Speech Audio</h3>
                        </div>
                        """
                    st.html(content)
                    
                    # Display the refined speech audio
                    st.audio(io.BytesIO(audio_bytes), format="audio/wav")
                
                    # Display the AI version of original speech
                    if st.session_state.get('ai_audio_bytes'):
                        st.subheader("AI Version of Original Speech")
                        try:
                            ai_audio_io = io.BytesIO(st.session_state.ai_audio_bytes)
                            ai_audio_io.seek(0)
                            st.audio(ai_audio_io, format='audio/mp3')
                            
                            # Create a container for download buttons
                            download_container = st.container()
                            
                            # Add download button for AI version
                            with download_container:
                                st.markdown(
                                    f'<a href="data:audio/mp3;base64,{base64.b64encode(st.session_state.ai_audio_bytes).decode()}" download="ai_version.mp3" style="text-decoration: none;">'
                                    '<button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%;">'
                                    'Download AI Version'
                                    '</button>'
                                    '</a>',
                                    unsafe_allow_html=True
                                )
                            
                            # Add download button for refined speech
                            with download_container:
                                st.markdown(
                                    f'<a href="data:audio/wav;base64,{audio_base64}" download="refined_speech.wav" style="text-decoration: none;">'
                                    '<button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%;">'
                                    'Download Refined Speech Audio'
                                    '</button>'
                                    '</a>',
                                    unsafe_allow_html=True
                                )
                        except Exception as e:
                            st.error(f"Error displaying AI audio: {str(e)}")
                            st.write("Debug info - Audio data length:", len(st.session_state.ai_audio_bytes) if st.session_state.get('ai_audio_bytes') else "No audio data")
            
            except Exception as e:
                # Decrement usage counter if processing failed
                if not st.session_state.get('is_authenticated', False):
                    st.session_state.usage_count -= 1
                status_container.error(f"An error occurred: {str(e)}")
                st.stop()
            
    # Display Results
    if st.session_state.get('results'):
        st.subheader("Analysis Results")
        
        # Display original audio if available
        if uploaded_file:
            st.subheader("Original Speech")
            st.audio(uploaded_file, format='audio/wav')
            
            # Display AI version if available
            if st.session_state.get('ai_audio_bytes'):
                st.subheader("AI Version of Original Speech")
                try:
                    # Create a new BytesIO object from the stored bytes
                    ai_audio_io = io.BytesIO(st.session_state.ai_audio_bytes)
                    ai_audio_io.seek(0)
                    
                    # Display the audio
                    st.audio(ai_audio_io, format='audio/mp3')
                    
                    # Add download button
                    st.download_button(
                        label="Download AI Version",
                        data=st.session_state.ai_audio_bytes,
                        file_name="ai_version.mp3",
                        mime="audio/mp3"
                    )
                except Exception as e:
                    st.error(f"Error displaying AI audio: {str(e)}")
                    st.write("Debug info - Audio data length:", len(st.session_state.ai_audio_bytes) if st.session_state.get('ai_audio_bytes') else "No audio data")
        
        # Display transcription
        if st.session_state.get('transcription'):
            st.subheader("Transcription")
            st.write(st.session_state.transcription)
        
        # Display analysis results
        results = st.session_state.results
        # ... existing code ...

def main():
    st.set_page_config(
        page_title="SpeechSmith Services",
        page_icon="🎤",
        layout="wide"
    )
    services()

if __name__ == "__main__":
    main()
