#!/usr/bin/env python3
"""
Dataset Generation Script for SimpleTTS Training
Uses NeuTTS to generate high-quality audio+text pairs for training SimpleTTS.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import soundfile as sf

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


class DatasetGenerator:
    """Generate TTS dataset using NeuTTS as teacher model."""

    def __init__(self, model_path: str = "neutss-air-BF16.gguf"):
        print("Initializing NeuTTS for dataset generation...")
        self.tts = NeuTTSAir(
            backbone_repo=model_path,
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )

        # Load reference audio
        ref_audio_path = "neutts-air/samples/dave.wav"
        ref_text_path = "neutts-air/samples/dave.txt"

        if not os.path.exists(ref_audio_path):
            print(f"Reference audio not found: {ref_audio_path}")
            sys.exit(1)

        print(f"Loading reference audio: {ref_audio_path}")
        self.ref_codes = self.tts.encode_reference(ref_audio_path)

        with open(ref_text_path, 'r', encoding='utf-8') as f:
            self.ref_text = f.read().strip()

        print("Dataset generator ready!")

    def generate_sample(self, text: str, output_dir: str, sample_id: int):
        """Generate a single audio+text sample."""
        try:
            # Generate audio
            audio = self.tts.infer(text, self.ref_codes, self.ref_text)

            # Save audio
            audio_path = os.path.join(output_dir, "audio", "wav", f"sample_{sample_id:04d}.wav")
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)
            sf.write(audio_path, audio, 24000)

            # Save text
            text_path = os.path.join(output_dir, "text", f"sample_{sample_id:04d}.txt")
            os.makedirs(os.path.dirname(text_path), exist_ok=True)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(text)

            return {
                "id": sample_id,
                "text": text,
                "audio_path": audio_path,
                "text_path": text_path,
                "duration": len(audio) / 24000.0  # seconds
            }

        except Exception as e:
            print(f"Error generating sample {sample_id}: {e}")
            return None

    def generate_dataset(self, output_dir: str, num_samples: int = 140):
        """Generate complete dataset with train/val/test splits."""

        # Diverse training sentences
        base_sentences = [
            # Greetings and basic phrases
            "Hello, how are you today?",
            "Good morning, I hope you're well.",
            "Thank you for your assistance.",
            "I'm sorry, I don't understand.",
            "Please speak more clearly.",

            # Questions and requests
            "Can you help me with this?",
            "What time is it now?",
            "Where is the nearest station?",
            "How much does this cost?",
            "Can I have more information?",

            # Statements and information
            "The weather is beautiful today.",
            "I need to go to the store.",
            "This is very important information.",
            "Please listen carefully to what I say.",
            "Let me explain the situation.",

            # Professional/business
            "I would like to schedule a meeting.",
            "Please review this document.",
            "The project is progressing well.",
            "We need to discuss this further.",
            "Thank you for your attention.",

            # Emotional expressions
            "I'm so happy to hear that!",
            "This is really frustrating.",
            "I'm very excited about this.",
            "That's wonderful news!",
            "I'm disappointed with the results.",

            # Instructions and directions
            "Turn left at the next intersection.",
            "Press the button to continue.",
            "Follow the instructions carefully.",
            "Wait for the signal to change.",
            "Enter your password now.",

            # Numbers and technical
            "The number is one two three four.",
            "My phone number is five five five one two three four.",
            "The temperature is twenty degrees Celsius.",
            "The time is three fifteen PM.",
            "The address is one two three Main Street.",

            # Longer sentences
            "I would like to book a table for two at eight o'clock this evening.",
            "The meeting has been rescheduled to next Tuesday at ten AM.",
            "Please make sure to bring all the necessary documents with you.",
            "The customer service representative will assist you shortly.",
            "We appreciate your patience and understanding in this matter.",

            # Conversational
            "That's interesting, can you tell me more about it?",
            "I completely agree with your point of view.",
            "Let me think about that for a moment.",
            "I'm not sure I understand what you mean.",
            "That makes perfect sense to me now.",
        ]

        # Generate variations by combining and modifying
        training_sentences = []
        for sentence in base_sentences:
            training_sentences.append(sentence)
            # Add some variations
            training_sentences.append(sentence.lower())
            training_sentences.append(sentence.upper()[:1] + sentence.lower()[1:])

        # Extend to reach desired number
        while len(training_sentences) < num_samples:
            # Add more variations
            for sentence in base_sentences[:10]:  # Use first 10 as base
                variation = sentence
                training_sentences.append(variation)

        training_sentences = training_sentences[:num_samples]

        print(f"Generating {len(training_sentences)} samples...")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)

        # Generate all samples
        dataset = []
        for i, text in enumerate(training_sentences):
            print(f"Generating sample {i+1}/{len(training_sentences)}: {text[:50]}...")
            sample = self.generate_sample(text, output_dir, i)
            if sample:
                dataset.append(sample)

        # Create train/val/test splits
        train_size = int(0.7 * len(dataset))  # 70% train
        val_size = int(0.2 * len(dataset))    # 20% val
        test_size = len(dataset) - train_size - val_size  # 10% test

        splits = {
            "train": dataset[:train_size],
            "validation": dataset[train_size:train_size + val_size],
            "test": dataset[train_size + val_size:]
        }

        # Save metadata
        metadata = {
            "total_samples": len(dataset),
            "splits": {
                "train": len(splits["train"]),
                "validation": len(splits["validation"]),
                "test": len(splits["test"])
            },
            "sample_rate": 24000,
            "reference_audio": "neutts-air/samples/dave.wav",
            "reference_text": self.ref_text,
            "generation_model": "NeuTTS-Air-GGUF"
        }

        # Save split files
        for split_name, split_data in splits.items():
            split_file = os.path.join(output_dir, f"{split_name}.json")
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print("\nDataset generation complete!")
        print(f"Total samples: {len(dataset)}")
        print(f"Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
        print(f"Dataset saved to: {output_dir}")

        return splits


def main():
    parser = argparse.ArgumentParser(description="Generate TTS dataset using NeuTTS")
    parser.add_argument("--output_dir", default="./tts_dataset",
                       help="Output directory for dataset")
    parser.add_argument("--num_samples", type=int, default=140,
                       help="Total number of samples to generate (default: 140)")
    parser.add_argument("--model_path", default="neutss-air-BF16.gguf",
                       help="Path to NeuTTS GGUF model")

    args = parser.parse_args()

    # Create dataset generator
    generator = DatasetGenerator(args.model_path)

    # Generate dataset
    generator.generate_dataset(args.output_dir, args.num_samples)

    print(f"\nDataset ready for SimpleTTS training!")
    print(f"Use: python simple_tts.py --mode train --data_file {args.output_dir}/train.json")


if __name__ == "__main__":
    main()