#!/usr/bin/env python3
"""
RAG + NeuTTS Air Demonstration Script
Uses Retrieval-Augmented Generation to select appropriate voice profiles
for text-to-speech synthesis.
"""

import argparse
import numpy as np
import soundfile as sf
import os
import sys
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import warnings

# Suppress TensorFlow warnings for cleaner output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# Add the local neutts-air directory to the path
sys.path.insert(0, 'neutts-air')

# Set espeak environment variables
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
os.environ['PHONEMIZER_ESPEAK_PATH'] = r'C:\Program Files\eSpeak NG'

try:
    from neuttsair.neutts import NeuTTSAir
except ImportError as e:
    print(f"Failed to import NeuTTSAir: {e}")
    print("Make sure you're running this from the correct directory with dependencies installed.")
    sys.exit(1)


class VoiceProfile:
    """Represents a voice profile with reference audio and metadata."""

    def __init__(self, name: str, description: str, ref_audio_path: str, ref_text_path: Optional[str] = None,
                 ref_text: Optional[str] = None, embedding: Optional[np.ndarray] = None, emotions: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.ref_audio_path = ref_audio_path
        self.ref_text_path = ref_text_path
        self.ref_text = ref_text
        self.embedding = embedding if embedding is not None else np.array([])
        self.emotions = emotions if emotions else ["neutral"]  # Default to neutral


class RAGTTS:
    """RAG-enhanced TTS system that selects voices based on context."""

    def __init__(self, backbone_path: str = "neutss-air-BF16.gguf"):
        # Initialize embedding model for RAG
        print("Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize emotion classifier
        print("Loading emotion classifier...")
        self.emotion_classifier = pipeline("text-classification",
                                         model="j-hartmann/emotion-english-distilroberta-base",
                                         return_all_scores=True)

        # Initialize TTS model
        print("Loading TTS model...")
        self.tts = NeuTTSAir(
            backbone_repo=backbone_path,
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )

        # Initialize voice database
        self.voice_profiles: List[VoiceProfile] = []
        self._initialize_voice_database()

    def _initialize_voice_database(self):
        """Initialize the voice profile database with sample voices and emotions."""
        profiles_data = [
            {
                "name": "dave_professional",
                "description": "A professional male voice, clear and authoritative. Good for business presentations, tutorials, and formal announcements.",
                "ref_audio": "neutts-air/samples/dave.wav",
                "ref_text_path": "neutts-air/samples/dave.txt",
                "emotions": ["neutral", "serious", "confident"]
            },
            {
                "name": "dave_casual",
                "description": "A casual male voice, relaxed and conversational. Suitable for informal chats, podcasts, and friendly interactions.",
                "ref_audio": "neutts-air/samples/dave.wav",
                "ref_text_path": "neutts-air/samples/dave.txt",
                "emotions": ["neutral", "relaxed", "casual"]
            },
            {
                "name": "dave_enthusiastic",
                "description": "An enthusiastic male voice, energetic and positive. Perfect for motivational content, advertisements, and exciting narratives.",
                "ref_audio": "neutts-air/samples/dave.wav",
                "ref_text_path": "neutts-air/samples/dave.txt",
                "emotions": ["happy", "excited", "enthusiastic"]
            },
            {
                "name": "jo_friendly",
                "description": "A friendly female voice, warm and approachable. Suitable for casual conversations, storytelling, and customer service.",
                "ref_audio": "neutts-air/samples/jo.wav",
                "ref_text_path": "neutts-air/samples/jo.txt",
                "emotions": ["neutral", "happy", "joy"]
            },
            {
                "name": "jo_professional",
                "description": "A professional female voice, articulate and composed. Good for corporate communications, training sessions, and formal dialogues.",
                "ref_audio": "neutts-air/samples/jo.wav",
                "ref_text_path": "neutts-air/samples/jo.txt",
                "emotions": ["neutral", "professional", "composed"]
            },
            # Toronto speech dataset samples for enhanced emotional variety
            {
                "name": "oaf_neutral",
                "description": "An older adult female voice with neutral emotion. Calm and steady, suitable for informative content and announcements.",
                "ref_audio": "toronto_speech_data/TESS Toronto emotional speech set data/OAF_neutral/OAF_keen_neutral.wav",
                "ref_text": "keen",
                "emotions": ["neutral"]
            },
            {
                "name": "oaf_sad",
                "description": "An older adult female voice expressing sadness. Melancholic and somber, appropriate for emotional storytelling or reflective narratives.",
                "ref_audio": "toronto_speech_data/TESS Toronto emotional speech set data/OAF_Sad/OAF_rain_sad.wav",
                "ref_text": "rain",
                "emotions": ["sad", "melancholy", "sorrow"]
            },
            {
                "name": "oaf_angry",
                "description": "An older adult female voice with anger. Intense and forceful, good for dramatic content or expressing frustration.",
                "ref_audio": "toronto_speech_data/TESS Toronto emotional speech set data/OAF_angry/OAF_beg_angry.wav",
                "ref_text": "beg",
                "emotions": ["angry", "frustrated", "intense"]
            },
            {
                "name": "yaf_happy",
                "description": "A young adult female voice full of happiness. Cheerful and energetic, perfect for positive messages and joyful narratives.",
                "ref_audio": "toronto_speech_data/TESS Toronto emotional speech set data/YAF_happy/YAF_book_happy.wav",
                "ref_text": "book",
                "emotions": ["happy", "joy", "cheerful"]
            },
            {
                "name": "yaf_fear",
                "description": "A young adult female voice expressing fear. Anxious and alarmed, suitable for suspenseful content or cautionary tales.",
                "ref_audio": "toronto_speech_data/TESS Toronto emotional speech set data/YAF_fear/YAF_kite_fear.wav",
                "ref_text": "kite",
                "emotions": ["fear", "anxious", "alarmed"]
            },
            {
                "name": "yaf_disgust",
                "description": "A young adult female voice showing disgust. Repulsed and disapproving, appropriate for critical commentary or expressing aversion.",
                "ref_audio": "toronto_speech_data/TESS Toronto emotional speech set data/YAF_disgust/YAF_cab_disgust.wav",
                "ref_text": "cab",
                "emotions": ["disgust", "repulsed", "disapproving"]
            }
        ]

        print("Building voice profile database...")
        for profile_data in profiles_data:
            # Create embedding from description
            embedding = self.embedder.encode(profile_data["description"])
            # Convert to numpy array
            embedding = np.array(embedding)
            profile = VoiceProfile(
                name=profile_data["name"],
                description=profile_data["description"],
                ref_audio_path=profile_data["ref_audio"],
                ref_text_path=profile_data.get("ref_text_path"),
                ref_text=profile_data.get("ref_text"),
                embedding=embedding,
                emotions=profile_data["emotions"]
            )
            self.voice_profiles.append(profile)

        print(f"Loaded {len(self.voice_profiles)} voice profiles.")

    def detect_emotion(self, text: str) -> str:
        """Detect the dominant emotion in the text."""
        try:
            results = self.emotion_classifier(text)
            # Get the emotion with highest score
            emotions = results[0] if results else []
            if emotions:
                best_emotion = max(emotions, key=lambda x: x['score'])
                return best_emotion['label'].lower()
        except Exception as e:
            print(f"Emotion detection failed: {e}")

        return "neutral"  # Default fallback

    def retrieve_voice(self, text: str) -> VoiceProfile:
        """Retrieve the most appropriate voice profile for the given text."""
        # Detect emotion in the text
        detected_emotion = self.detect_emotion(text)
        print(f"Detected emotion: {detected_emotion}")

        # Create embedding for input text
        text_embedding = self.embedder.encode(text)
        text_embedding = np.array(text_embedding)  # Convert to numpy

        # Calculate similarities with emotion bonus
        similarities = []
        for profile in self.voice_profiles:
            # Base semantic similarity
            semantic_sim = cosine_similarity(
                text_embedding.reshape(1, -1),
                profile.embedding.reshape(1, -1)
            )[0][0]

            # Emotion matching bonus
            emotion_bonus = 0.3 if detected_emotion in profile.emotions else 0.0

            # Combined score
            total_score = semantic_sim + emotion_bonus
            similarities.append((profile, total_score, semantic_sim, emotion_bonus))

        # Return the most similar profile
        best_profile, best_score, semantic_sim, emotion_bonus = max(similarities, key=lambda x: x[1])
        print(".3f")
        if emotion_bonus > 0:
            print(f"  Emotion bonus: +{emotion_bonus} (detected: {detected_emotion})")
        return best_profile

    def generate_speech(self, text: str, output_path: str = "rag_output.wav") -> str:
        """Generate speech using RAG to select the appropriate voice."""
        print(f"Input text: {text}")

        # Retrieve appropriate voice
        selected_voice = self.retrieve_voice(text)

        # Load reference text
        if selected_voice.ref_text_path:
            try:
                with open(selected_voice.ref_text_path, 'r', encoding='utf-8') as f:
                    ref_text = f.read().strip()
            except FileNotFoundError:
                print(f"Warning: Reference text file not found: {selected_voice.ref_text_path}")
                ref_text = "Hello world"  # Fallback
        else:
            ref_text = selected_voice.ref_text or "Hello world"

        # Encode reference audio
        print(f"Encoding reference audio: {selected_voice.ref_audio_path}")
        ref_codes = self.tts.encode_reference(selected_voice.ref_audio_path)

        # Generate speech
        print("Generating speech...")
        wav = self.tts.infer(text, ref_codes, ref_text)

        # Save audio
        sf.write(output_path, wav, 24000)
        print(f"Audio saved to: {output_path}")

        return output_path


def main():
    parser = argparse.ArgumentParser(description="RAG-enhanced NeuTTS Air demonstration")
    parser.add_argument("--text", help="Text to synthesize")
    parser.add_argument("--output", default="rag_output.wav", help="Output audio file")
    parser.add_argument("--backbone", default="neutss-air-BF16.gguf", help="Path to GGUF backbone model")

    args = parser.parse_args()

    # Initialize RAG-TTS system
    rag_tts = RAGTTS(backbone_path=args.backbone)

    # Get text input
    if not args.text:
        args.text = input("Enter the text to synthesize: ")

    # Generate speech
    output_file = rag_tts.generate_speech(args.text, args.output)

    print(f"\nRAG-TTS generation complete!")
    print(f"Output file: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    main()