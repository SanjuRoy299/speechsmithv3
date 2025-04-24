import streamlit as st
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from groq import Groq
import torch
from transformers import pipeline
import wave
import contextlib
from textblob import TextBlob
import speech_recognition as sr
import tempfile
import os
import json
import re
from dotenv import load_dotenv
import markdown
from typing import Optional
load_dotenv()

class SpeechAnalyzer:
    def __init__(self):
        # Initialize Groq client
        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
    
    def get_audio_duration(self, audio_path):
        """Get duration of audio file"""
        with contextlib.closing(wave.open(audio_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        return duration
    
    def transcribe_audio(self, audio_path):
        """Convert speech to text using OpenAI's Whisper API"""
        try:
            import openai
            from openai import OpenAI
            
            # Initialize OpenAI client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Open the audio file
            with open(audio_path, "rb") as audio_file:
                # Transcribe the audio file
                st.write("Transcribing using OpenAI Whisper...")
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    response_format="text"
                )
                
                # Get the transcription
                transcription = response.strip()
                
                # Verify the transcription
                if not transcription:
                    st.error("No speech detected in the audio file")
                    return None
                
                # Print transcription length for debugging
                #st.write(f"Transcription length: {len(transcription)} characters")
                
                return transcription
                
        except Exception as e:
            st.error(f"Error in transcription: {str(e)}")
            return None

    def analyze_text_with_llama(self, text, analysis_type, topic=None):
        """Analyze text using Llama 3 model via Groq"""
        prompts = {
            'pronunciation': f"""
                Analyze the following text for pronunciation accuracy and identify any potentially mispronounced words:
                
                Text: {text}
                
                Please provide a JSON object with the following structure:
                {{
                    "mispronounced_words": {{
                        "word1": confidence_score_1,  # Score between 0 and 1, where 1 is perfect pronunciation
                        "word2": confidence_score_2
                    }},
                    "pronunciation_guidance": {{
                        "word1": "correct pronunciation guide",
                        "word2": "correct pronunciation guide"
                    }}
                }}
                
                Guidelines:
                1. Only include words that are likely to be mispronounced
                2. Consider common pronunciation challenges for non-native speakers
                3. Focus on words with complex syllable structures or unusual spellings
                4. Assign confidence scores based on pronunciation difficulty
                5. Provide clear pronunciation guidance for each identified word
                
                Only return the JSON object, nothing else.
            """,
            'mood': f"""
                Analyze the mood and emotional content of the following text:
                Text: {text}
                Topic: {topic}
                
                Please provide a JSON object with the following structure:
                {{
                    "primary_emotion": "emotion name",
                    "secondary_emotions": ["emotion1", "emotion2"],
                    "intensity": 0.0-1.0,
                    "formality": "formality level",
                    "audience_suitability": "suitability assessment",
                    "mood_suitability_assessment": {{
                        "assessment": "suitability assessment",
                        "reasons": ["reason1", "reason2"]
                    }}
                }}
                
                Only return the JSON object, nothing else.
            """,
            'filler_analysis': f"""
                Analyze the following text for filler words and speech patterns:
                Text: {text}
                
                Please provide a JSON object with the following structure:
                {{
                    "filler_words": {{
                        "word1": count1,
                        "word2": count2
                    }},
                    "total_words": total_word_count,
                    "style_assessment": "assessment text",
                    "suggestions": ["suggestion1", "suggestion2"]
                }}
                
                Only return the JSON object, nothing else.
            """
        }

        try:
            completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional speech analyst. Provide analysis in valid JSON format only."
                    },
                    {
                        "role": "user",
                        "content": prompts[analysis_type]
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            
            # Extract and clean the response content
            content = completion.choices[0].message.content.strip()
            
            # Remove any potential extra text before or after the JSON
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]
            
            # Parse JSON response
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {str(e)}")
                print(f"Response content: {content}")
                return None
                
        except Exception as e:
            print(f"Error in Llama analysis: {str(e)}")
            return None

    def analyze_pronunciation(self, audio_path, transcript):
        """Analyze pronunciation using audio features and Llama"""
        try:
            # Get basic audio features
            y, sr = librosa.load(audio_path)
            
            # Get Llama analysis for pronunciation
            llama_analysis = self.analyze_text_with_llama(transcript, 'pronunciation')

            if llama_analysis:
                # Calculate overall accuracy based on mispronounced words
                difficult_words = llama_analysis.get('mispronounced_words', {})
                total_words = len(transcript.split())
                mispronounced_count = len(difficult_words)
                
                if total_words > 0:
                    accuracy = round((1 - (mispronounced_count / total_words)) * 100, 2)
                else:
                    accuracy = 0

                # Generate feedback based on analysis
                feedback = []
                if mispronounced_count > 0:
                    if mispronounced_count / total_words > 0.2:  # More than 20% mispronounced
                        feedback.append("The speech contains several pronunciation challenges that affect clarity.")
                    else:
                        feedback.append("The speech has some pronunciation issues but remains generally understandable.")
                    
                    if any(score < 0.5 for score in difficult_words.values()):
                        feedback.append("Some words have very low confidence scores, indicating significant pronunciation difficulties.")
                else:
                    feedback.append("Pronunciation is clear and accurate.")

                return {
                    'accuracy': accuracy,
                    'feedback': ' '.join(feedback),
                    'difficult_words': difficult_words,
                    'pronunciation_guidance': llama_analysis.get('pronunciation_guidance', {})
                }
            else:
                return {
                    'accuracy': 0,
                    'feedback': 'Unable to analyze pronunciation',
                    'difficult_words': {},
                    'pronunciation_guidance': {}
                }
                
        except Exception as e:
            print(f"Error in pronunciation analysis: {str(e)}")
            return {
                'accuracy': 0,
                'feedback': f'Error in pronunciation analysis: {str(e)}',
                'difficult_words': {},
                'pronunciation_guidance': {}
            }

    def analyze_pitch(self, audio_path, gender):
        """Analyze pitch characteristics"""
        try:
            if gender not in ['male', 'female']:
                return {
                    'variation': 'Gender not specified for pitch analysis',
                    'consistency': 'Gender not specified for pitch analysis',
                    'average': 0.0
                }
            
            # Load audio file
            y, sr = librosa.load(audio_path)
            
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get non-zero pitches
            pitches_flat = pitches[magnitudes > np.max(magnitudes) * 0.1]
            pitches_flat = pitches_flat[pitches_flat > 0]
            
            if len(pitches_flat) == 0:
                return {
                    'variation': 'Unable to detect pitch',
                    'consistency': 'Unable to detect pitch',
                    'average': 0.0
                }
            
            # Calculate pitch statistics
            pitch_mean = np.mean(pitches_flat)
            pitch_std = np.std(pitches_flat)
            pitch_range = np.max(pitches_flat) - np.min(pitches_flat)
            
            # Define gender-specific thresholds
            thresholds = {
                'male': {
                    'monotonous': 20,
                    'erratic': 80,
                    'normal_range': (85, 180),
                    'variation_threshold': 40
                },
                'female': {
                    'monotonous': 30,
                    'erratic': 100,
                    'normal_range': (165, 255),
                    'variation_threshold': 50
                }
            }
            
            current_threshold = thresholds[gender]
            
            # Analyze pitch variation
            if pitch_std < current_threshold['monotonous']:
                variation = 'Limited pitch variation. Consider adding more vocal variety to maintain audience interest.'
            elif pitch_std > current_threshold['erratic']:
                variation = 'Excessive pitch variation. Try to maintain more consistent pitch levels.'
            else:
                variation = 'Good pitch variation. Maintains audience interest while being clear and consistent.'
            
            # Analyze pitch consistency
            if pitch_range < current_threshold['variation_threshold']:
                consistency = 'Pitch remains too consistent. Consider varying your pitch more to emphasize key points.'
            elif pitch_range > current_threshold['variation_threshold'] * 2:
                consistency = 'Pitch varies too much. Try to maintain more consistent levels while still emphasizing important points.'
            else:
                consistency = 'Pitch consistency is good. Maintains appropriate variation while staying clear.'
            
            # Check if average pitch is within normal range
            normal_min, normal_max = current_threshold['normal_range']
            if pitch_mean < normal_min:
                consistency += ' Average pitch is lower than typical for your gender.'
            elif pitch_mean > normal_max:
                consistency += ' Average pitch is higher than typical for your gender.'
            
            return {
                'variation': variation,
                'consistency': consistency,
                'average': float(pitch_mean),
                'standard_deviation': float(pitch_std),
                'range': float(pitch_range)
            }
            
        except Exception as e:
            print(f"Error in pitch analysis: {str(e)}")
            return {
                'variation': f'Error in pitch analysis: {str(e)}',
                'consistency': f'Error in pitch analysis: {str(e)}',
                'average': 0.0
            }

    def analyze_speech_rate(self, audio_path, transcript):
        """Analyze speech rate and pauses"""
        try:
            # Get audio duration
            duration = self.get_audio_duration(audio_path)
            if duration == 0:
                return {
                    'wpm': 0,
                    'assessment': 'Unable to calculate speech rate: Audio duration is zero',
                    'filler_words': {}
                }

            # Load audio and detect speech segments
            y, sr = librosa.load(audio_path)
            intervals = librosa.effects.split(y, top_db=20)
            speech_duration = sum(interval[1] - interval[0] for interval in intervals) / sr
            
            # Get filler word analysis
            filler_analysis = self.analyze_text_with_llama(transcript, 'filler_analysis')
            
            # Calculate word counts
            total_words = len(transcript.split())
            filler_words = filler_analysis.get('filler_words', {}) if filler_analysis else {}
            filler_count = sum(filler_words.values())
            effective_words = total_words - filler_count
            
            # Calculate words per minute
            words_per_minute = (effective_words / duration) * 60
            
            # Generate assessment based on WPM
            if words_per_minute < 120:
                assessment = "Speaking rate is slow. Consider increasing your pace for better engagement."
            elif words_per_minute > 160:
                assessment = "Speaking rate is fast. Consider slowing down for better clarity."
            else:
                assessment = "Speaking rate is optimal for clear communication."
            
            return {
                'wpm': round(words_per_minute, 2),
                'assessment': assessment,
                'filler_words': filler_words,
                'total_words': total_words,
                'effective_words': effective_words,
                'speech_duration': round(speech_duration, 2),
                'total_duration': round(duration, 2)
            }
            
        except Exception as e:
            print(f"Error in speech rate analysis: {str(e)}")
            return {
                'wpm': 0,
                'assessment': f'Error in speech rate analysis: {str(e)}',
                'filler_words': {}
            }

    def analyze_mood(self, transcript, topic):
        """Analyze speech mood using Llama"""
        mood_analysis = self.analyze_text_with_llama(transcript,  'mood', topic)
        return mood_analysis

def analyze_speaking_style(rate_analysis):
    """Analyze speaking style based on rate and filler words"""
    wpm = rate_analysis['wpm']
    assessment = rate_analysis['assessment']
    filler_ratio = rate_analysis['filler_count'] / rate_analysis['total_words'] if rate_analysis['total_words'] > 0 else 0
    
    style_feedback = []
    
    # Speech rate analysis
    if wpm < 120:
        style_feedback.append("Speaking rate is slow (below 120 words per minute). Consider increasing your pace.")
    elif wpm > 160:
        style_feedback.append("Speaking rate is fast (above 160 words per minute). Consider slowing down.")
    else:
        style_feedback.append("Speaking rate is at a good pace (120-160 words per minute).")
    
    # Assessment analysis
    style_feedback.append(assessment)
    
    # Filler words analysis
    if filler_ratio > 0.1:
        style_feedback.append(f"High usage of filler words ({filler_ratio:.1%}). Work on reducing them.")
    else:
        style_feedback.append(f"Good control over filler words ({filler_ratio:.1%}).")
    
    # Word count analysis
    duration_minutes = rate_analysis['total_duration'] / 60
    ideal_word_count = int(duration_minutes * 140)  # assuming 140 wpm is ideal
    actual_words = rate_analysis['total_words']
    
    if actual_words < ideal_word_count * 0.8:
        style_feedback.append(f"Content is too brief for the duration. Consider adding more content (ideal: {ideal_word_count} words).")
    elif actual_words > ideal_word_count * 1.2:
        style_feedback.append(f"Content is too long for the duration. Consider reducing content (ideal: {ideal_word_count} words).")
    else:
        style_feedback.append(f"Good content length for the duration ({actual_words} words).")
    
    return style_feedback


def generate_feedback(analyzer_results, topic):
    """Generate comprehensive feedback based on actual analysis results"""
    feedback = {
        'tone_and_style': {
            'title': 'Tone and Style Analysis',
            'original': [],
            'revised': []
        },
        'pacing_and_timing': {
            'title': 'Pacing and Timing Analysis',
            'original': [],
            'revised': []
        },
        'clarity_and_structure': {
            'title': 'Clarity and Structure Evaluation',
            'original': [],
            'revised': []
        },
        'technical_analysis': {
            'title': 'Technical Analysis Results',
            'pronunciation': [],
            'speech_rate': [],
            'mood': [],
            'pitch': []
        },
        'improvement_recommendations': {
            'title': 'Specific Improvement Recommendations',
            'original': [],
            'additional': []
        }
    }
    
    # Tone and Style Analysis based on mood analysis
    if analyzer_results.get('mood'):
        mood = analyzer_results['mood']
        feedback['tone_and_style']['original'].extend([
            f"Primary emotion: {mood.get('primary_emotion', 'Not analyzed')}",
            f"Formality level: {mood.get('formality', 'Not analyzed')}",
            f"Audience suitability: {mood.get('audience_suitability', 'Not analyzed')}"
        ])
        
        if mood.get('mood_suitability_assessment'):
            assessment = mood['mood_suitability_assessment']
            feedback['tone_and_style']['revised'].extend([
                f"Assessment: {assessment.get('assessment', 'Not analyzed')}",
                "Reasons:"
            ])
            if isinstance(assessment.get('reasons'), list):
                feedback['tone_and_style']['revised'].extend(assessment['reasons'])

    # Pacing and Timing Analysis based on speech rate
    if analyzer_results.get('speech_rate'):
        rate = analyzer_results['speech_rate']
        feedback['pacing_and_timing']['original'].extend([
            f"Words per minute: {rate.get('wpm', 0)}",
            f"Assessment: {rate.get('assessment', 'Not analyzed')}",
            f"Speech duration: {rate.get('speech_duration', 0)} seconds",
            f"Total duration: {rate.get('total_duration', 0)} seconds"
        ])
        
        if rate.get('filler_words'):
            feedback['pacing_and_timing']['revised'].append("Filler words analysis:")
            for word, count in rate['filler_words'].items():
                feedback['pacing_and_timing']['revised'].append(f"- '{word}': {count} occurrences")

    # Clarity and Structure based on pronunciation
    if analyzer_results.get('pronunciation'):
        pron = analyzer_results['pronunciation']
        feedback['clarity_and_structure']['original'].extend([
            f"Overall accuracy: {pron.get('accuracy', 0)}%",
            f"Accuracy feedback: {pron.get('feedback', 'Not analyzed')}"
        ])
        
        if pron.get('difficult_words'):
            feedback['clarity_and_structure']['revised'].append("Difficult words:")
            for word, score in pron['difficult_words'].items():
                feedback['clarity_and_structure']['revised'].append(f"- {word}: {score:.2f}")

    # Technical Analysis
    if analyzer_results.get('pronunciation'):
        pron = analyzer_results['pronunciation']
        feedback['technical_analysis']['pronunciation'].extend([
            f"Overall Accuracy: {pron.get('accuracy', 0)}%",
            f"Feedback: {pron.get('feedback', 'Not analyzed')}"
        ])
        if pron.get('difficult_words'):
            feedback['technical_analysis']['pronunciation'].append("Mispronounced Words:")
            for word, score in pron['difficult_words'].items():
                feedback['technical_analysis']['pronunciation'].append(f"- {word} (confidence: {score:.2f})")

    if analyzer_results.get('speech_rate'):
        rate = analyzer_results['speech_rate']
        feedback['technical_analysis']['speech_rate'].extend([
            f"Words per minute: {rate.get('wpm', 0)}",
            f"Assessment: {rate.get('assessment', 'Not analyzed')}",
            "Filler Words Usage:"
        ])
        if rate.get('filler_words'):
            for word, count in rate['filler_words'].items():
                feedback['technical_analysis']['speech_rate'].append(f"- '{word}': {count} occurrences")

    if analyzer_results.get('mood'):
        mood = analyzer_results['mood']
        feedback['technical_analysis']['mood'].extend([
            f"Primary Emotion: {mood.get('primary_emotion', 'Not analyzed')}",
            f"Formality: {mood.get('formality', 'Not analyzed')}",
            f"Audience Suitability: {mood.get('audience_suitability', 'Not analyzed')}"
        ])
        if mood.get('mood_suitability_assessment'):
            assessment = mood['mood_suitability_assessment']
            feedback['technical_analysis']['mood'].extend([
                f"Assessment: {assessment.get('assessment', 'Not analyzed')}",
                "Reasons:"
            ])
            if isinstance(assessment.get('reasons'), list):
                feedback['technical_analysis']['mood'].extend(assessment['reasons'])

    if analyzer_results.get('pitch'):
        pitch = analyzer_results['pitch']
        feedback['technical_analysis']['pitch'].extend([
            f"Variation: {pitch.get('variation', 'Not analyzed')}",
            f"Consistency: {pitch.get('consistency', 'Not analyzed')}",
            f"Average Pitch: {pitch.get('average', 0):.1f} Hz"
        ])

    # Improvement Recommendations based on all analyses
    if analyzer_results.get('pronunciation'):
        pron = analyzer_results['pronunciation']
        if pron.get('difficult_words'):
            feedback['improvement_recommendations']['original'].append("Pronunciation improvements needed for:")
            for word in pron['difficult_words'].keys():
                feedback['improvement_recommendations']['original'].append(f"- {word}")

    if analyzer_results.get('speech_rate'):
        rate = analyzer_results['speech_rate']
        if rate.get('filler_words'):
            feedback['improvement_recommendations']['additional'].append("Reduce usage of filler words:")
            for word in rate['filler_words'].keys():
                feedback['improvement_recommendations']['additional'].append(f"- {word}")

    if analyzer_results.get('pitch'):
        pitch = analyzer_results['pitch']
        if pitch.get('variation') == 'Good pitch variation in voice':
            feedback['improvement_recommendations']['additional'].append("Maintain current pitch variation")
        else:
            feedback['improvement_recommendations']['additional'].append("Work on improving pitch variation")

    return feedback

def format_feedback_to_html(feedback, transcription, refined_speech):
    """Format feedback into structured HTML"""
    
    def format_section(items):
        if not items:
            return "<p>No feedback available</p>"
        
        html = "<ul>"
        for item in items:
            html += f"<li>{item}</li>"
        html += "</ul>"
        return html

    content = """
        <div class="feedback-container">
            <h1>Speech Analysis Feedback</h1>
    """

    # Format main sections
    for section_key, section_data in feedback.items():
        if section_key != 'technical_analysis':
            content += f"""
                <section class="feedback-section">
                    <h2>{section_data['title']}</h2>
            """
            
            if 'original' in section_data:
                content += f"""
                    <div class="subsection">
                        <h3>Original Speech:</h3>
                        {format_section(section_data['original'])}
                    </div>
                    <div class="subsection">
                        <h3>Revised Speech:</h3>
                        {format_section(section_data['revised'])}
                    </div>
                """
            else:
                content += format_section(section_data)
            
            content += "</section>"
        else:
            # Technical Analysis section
            content += f"""
                <section class="feedback-section">
                    <h2>{section_data['title']}</h2>
            """
            
            # Pronunciation Analysis
            if section_data['pronunciation']:
                content += """
                    <div class="subsection">
                        <h3>Pronunciation Analysis:</h3>
                        <ul>
                """
                for item in section_data['pronunciation']:
                    content += f"<li>{item}</li>"
                content += "</ul></div>"

            # Speech Rate Analysis
            if section_data['speech_rate']:
                content += """
                    <div class="subsection">
                        <h3>Speech Rate Analysis:</h3>
                        <ul>
                """
                for item in section_data['speech_rate']:
                    content += f"<li>{item}</li>"
                content += "</ul></div>"

            # Mood Analysis
            if section_data['mood']:
                content += """
                    <div class="subsection">
                        <h3>Mood Analysis:</h3>
                        <ul>
                """
                for item in section_data['mood']:
                    content += f"<li>{item}</li>"
                content += "</ul></div>"

            # Pitch Analysis
            if section_data['pitch']:
                content += """
                    <div class="subsection">
                        <h3>Pitch Analysis:</h3>
                        <ul>
                """
                for item in section_data['pitch']:
                    content += f"<li>{item}</li>"
                content += "</ul></div>"
            
            content += "</section>"

    content += """
        </div>
        <style>
            .feedback-container {
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                font-family: Arial, sans-serif;
            }
            .feedback-section {
                margin-bottom: 30px;
                background-color: white;
                border-radius: 5px;
                padding: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .subsection {
                margin-left: 20px;
                margin-bottom: 20px;
                border-left: 3px solid #3498db;
                padding-left: 15px;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }
            h2 {
                color: #34495e;
                margin-top: 25px;
                font-size: 1.5em;
            }
            h3 {
                color: #7f8c8d;
                margin-top: 15px;
                font-size: 1.2em;
            }
            ul {
                list-style-type: none;
                padding-left: 20px;
                margin-top: 10px;
            }
            li {
                margin: 8px 0;
                position: relative;
                line-height: 1.4;
            }
            li:before {
                content: "•";
                color: #3498db;
                font-weight: bold;
                position: absolute;
                left: -15px;
            }
            .technical-item {
                background-color: #f8f9fa;
                padding: 8px 12px;
                border-radius: 4px;
                margin: 5px 0;
            }
        </style>
    """
    
    return content

def main():
    st.title("Advanced Speech Analysis Tool")
    
    # Initialize analyzer
    analyzer = SpeechAnalyzer()
    
    # User inputs
    topic = st.text_input("Topic of Speech")
    audio_file = st.file_uploader("Upload Audio File (.wav format)", type=['wav'])
    gender = st.radio("Select Gender (for pitch analysis)", ['male', 'female'])
    
    if audio_file and topic:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            audio_path = tmp_file.name
        
        try:
            # Process audio
            with st.spinner("Analyzing speech..."):
                # Get transcript
                transcript = analyzer.transcribe_audio(audio_path)
                
                if transcript:
                    # Display Original Audio
                    st.header("Original Speech")
                    st.audio(audio_file, format='audio/wav')
                    
                    # Perform analysis
                    results = {
                        'pronunciation': analyzer.analyze_pronunciation(audio_path, transcript),
                        'pitch': analyzer.analyze_pitch(audio_path, gender),
                        'speech_rate': analyzer.analyze_speech_rate(audio_path, transcript),
                        'mood': analyzer.analyze_mood(transcript, topic)
                    }
                    
                    # Generate feedback
                    feedback = generate_feedback(results, topic)
                    
                    # Format and display feedback
                    formatted_feedback = format_feedback_to_html(feedback, transcript, None)
                    st.markdown(formatted_feedback, unsafe_allow_html=True)
                    
                    # Add custom CSS for better styling
                    st.markdown("""
                        <style>
                            .feedback-container {
                                background-color: #f8f9fa;
                                border-radius: 10px;
                                padding: 25px;
                                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                            }
                            .feedback-section {
                                background-color: white;
                                border-radius: 5px;
                                padding: 15px;
                                margin-bottom: 20px;
                                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
                            }
                            .subsection {
                                border-left: 3px solid #3498db;
                                padding-left: 15px;
                                margin: 10px 0;
                            }
                            li {
                                padding: 5px 0;
                            }
                        </style>
                    """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        finally:
            # Clean up temporary file
            os.unlink(audio_path)

if __name__ == "__main__":
    main()
