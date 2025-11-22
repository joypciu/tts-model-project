#!/usr/bin/env python3
"""
Knowledge Distillation Training for SimpleTTS
Trains SimpleTTS (student) to learn from NeuTTS (teacher) through distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import argparse
from tqdm import tqdm
from typing import Optional
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
    from simple_tts import SimpleTTS, TTSDataset
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    sys.exit(1)


class DistillationTrainer:
    """Knowledge distillation trainer for SimpleTTS learning from NeuTTS."""

    def __init__(self, teacher_model_path: str = "neutss-air-BF16.gguf",
                 student_model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device)

        # Initialize teacher model (NeuTTS)
        print("Loading teacher model (NeuTTS)...")
        self.teacher = NeuTTSAir(
            backbone_repo=teacher_model_path,
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )

        # Load reference audio for teacher
        ref_audio_path = "neutts-air/samples/dave.wav"
        ref_text_path = "neutts-air/samples/dave.txt"

        if not os.path.exists(ref_audio_path):
            print(f"Reference audio not found: {ref_audio_path}")
            sys.exit(1)

        print(f"Loading reference audio: {ref_audio_path}")
        self.teacher_ref_codes = self.teacher.encode_reference(ref_audio_path)

        with open(ref_text_path, 'r', encoding='utf-8') as f:
            self.teacher_ref_text = f.read().strip()

        # Initialize student model (SimpleTTS)
        print("Loading student model (SimpleTTS)...")
        self.student = SimpleTTS(model_path=student_model_path, device=device)

        print("Distillation trainer ready!")

    def generate_teacher_output(self, text: str):
        """Generate output from teacher model (NeuTTS)."""
        try:
            # Generate audio using NeuTTS
            audio = self.teacher.infer(text, self.teacher_ref_codes, self.teacher_ref_text)

            # Extract mel spectrogram (target for student)
            import librosa
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=24000,
                n_fft=1024,
                hop_length=256,
                n_mels=80,
                fmin=0,
                fmax=8000
            )
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

            return torch.tensor(mel_spec, dtype=torch.float32), audio

        except Exception as e:
            print(f"Teacher generation failed for '{text}': {e}")
            return None, None

    def distillation_loss(self, student_output, teacher_mel, temperature=1.0):
        """Calculate distillation loss between student and teacher."""
        # Detach teacher to prevent gradient flow
        teacher_mel = teacher_mel.detach()

        # Resize student output to match teacher dimensions
        if student_output.shape != teacher_mel.shape:
            student_output = F.interpolate(
                student_output.unsqueeze(0),
                size=teacher_mel.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        # L1 loss for spectrogram matching (simpler than KL divergence)
        loss = F.l1_loss(student_output, teacher_mel)

        return loss

    def train_distillation(self, texts: list, epochs: int = 5, batch_size: int = 4,
                          learning_rate: float = 1e-4, save_path: str = "./distilled_simple_tts.pth"):
        """Train student model using distillation from teacher."""

        # Prepare text data
        text_data = [{"text": text, "audio_path": None} for text in texts]

        # Create dataset and dataloader
        dataset = TTSDataset.__new__(TTSDataset)  # Create without loading files
        dataset.data = text_data
        dataset.max_length = 256
        dataset.sample_rate = 22050

        # Custom collate for distillation
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            return {"texts": texts}

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Optimizer
        optimizer = torch.optim.Adam(self.student.model.parameters(), lr=learning_rate)

        print(f"Starting distillation training for {epochs} epochs...")
        print(f"Teacher: NeuTTS, Student: SimpleTTS")
        print(f"Training on {len(texts)} text samples")

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_texts = batch["texts"]
                batch_loss = 0

                for text in batch_texts:
                    # Generate teacher output
                    teacher_mel, teacher_audio = self.generate_teacher_output(text)
                    if teacher_mel is None:
                        continue

                    # Generate student output
                    tokens = self.student.text_to_tokens(text)
                    tokens = tokens + [0] * (256 - len(tokens))
                    text_tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)
                    text_lengths = torch.tensor([len(text)]).to(self.device)

                    # Forward pass through student (with gradients)
                    student_outputs = self.student.model(text_tokens, text_lengths)
                    student_mel = self.student.model._prosody_to_mel(student_outputs, 256, 80, len(text))
                    student_audio = self.student.model.vocoder(student_mel, 0)  # Generate audio

                    # Calculate distillation loss directly on audio waveforms
                    teacher_audio_tensor = torch.tensor(teacher_audio, dtype=torch.float32).to(self.device)
                    if student_audio.shape != teacher_audio_tensor.shape:
                        # Resize student audio to match teacher
                        student_audio = torch.nn.functional.interpolate(
                            student_audio.unsqueeze(0), size=len(teacher_audio), mode='linear'
                        ).squeeze(0)

                    # Use L1 loss on audio waveforms
                    loss = torch.nn.functional.l1_loss(student_audio, teacher_audio_tensor)

                    # Backward pass (only through student parameters)
                    optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.student.model.parameters(), max_norm=1.0)
                    optimizer.step()

                    batch_loss += loss.item()

                epoch_loss += batch_loss / len(batch_texts)
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(".4f")

        # Save distilled model
        self.student.save_model(save_path)
        print(f"Distillation training completed! Model saved to {save_path}")
        print("SimpleTTS has learned from NeuTTS through knowledge distillation!")

    def evaluate_distillation(self, test_texts: list):
        """Evaluate the distilled model's performance."""
        print("Evaluating distilled model...")

        results = []
        for text in test_texts:
            # Generate with both models
            teacher_mel, teacher_audio = self.generate_teacher_output(text)
            if teacher_mel is None:
                continue

            # Student generation
            tokens = self.student.text_to_tokens(text)
            tokens = tokens + [0] * (256 - len(tokens))
            text_tokens = torch.tensor([tokens], dtype=torch.long).to(self.device)
            text_lengths = torch.tensor([len(text)]).to(self.device)

            student_audio = self.student.model.generate_audio(text_tokens, text_lengths)
            # Ensure we have a numpy array, not a generator
            if hasattr(student_audio, '__iter__') and not isinstance(student_audio, np.ndarray):
                # If it's a generator, convert to array
                student_audio = np.concatenate(list(student_audio))

            # Calculate similarity metrics
            # (Simplified: just check if both generated successfully)
            # Get audio lengths safely
            teacher_length = 0
            if teacher_audio is not None:
                try:
                    teacher_length = len(teacher_audio)
                except:
                    teacher_length = 0

            student_length = 0
            if student_audio is not None:
                try:
                    student_length = len(student_audio)
                except:
                    student_length = 0

            results.append({
                "text": text,
                "teacher_audio_length": teacher_length,
                "student_audio_length": student_length
            })

        print(f"Evaluation complete. Generated audio for {len(results)}/{len(test_texts)} test samples")
        return results


def create_training_texts(num_samples: int = 100):
    """Create diverse training texts for distillation."""
    base_texts = [
        "Hello, how are you today?",
        "Thank you for your help.",
        "I need assistance with this.",
        "Please listen carefully.",
        "This is very important.",
        "Welcome to our service.",
        "Let me explain the process.",
        "I'm here to help you.",
        "Please provide more details.",
        "That sounds like a great idea.",
        "I completely agree with you.",
        "Let me think about that.",
        "I'm not sure I understand.",
        "That makes perfect sense.",
        "The weather is beautiful today.",
        "I need to go to the store.",
        "This is an important message.",
        "Please follow the instructions.",
        "Wait for the signal to change.",
        "Enter your password now.",
        "The number is one two three four.",
        "My phone number is five five five one two three four.",
        "The temperature is twenty degrees Celsius.",
        "The time is three fifteen PM.",
        "The address is one two three Main Street.",
        "I would like to book a table for two.",
        "The meeting has been rescheduled.",
        "Please review this document.",
        "The project is progressing well.",
        "We need to discuss this further.",
        "I'm so happy to hear that!",
        "This is really frustrating.",
        "I'm very excited about this.",
        "That's wonderful news!",
        "I'm disappointed with the results.",
        "Turn left at the next intersection.",
        "Press the button to continue.",
        "Follow the instructions carefully.",
        "Wait for the signal to change.",
        "Enter your password now.",
    ]

    # Generate variations
    training_texts = []
    for text in base_texts:
        training_texts.append(text)
        training_texts.append(text.lower())
        training_texts.append(text.upper()[:1] + text.lower()[1:])

    # Extend to desired number
    while len(training_texts) < num_samples:
        for text in base_texts[:20]:
            training_texts.append(text)

    return training_texts[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Training for SimpleTTS")
    parser.add_argument("--teacher_model", default="neutss-air-BF16.gguf",
                       help="Path to NeuTTS teacher model")
    parser.add_argument("--student_model", default=None,
                       help="Path to existing SimpleTTS student model")
    parser.add_argument("--output_model", default="./distilled_simple_tts.pth",
                       help="Path to save distilled model")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of distillation epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for distillation")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation after training")

    args = parser.parse_args()

    # Create distillation trainer
    trainer = DistillationTrainer(
        teacher_model_path=args.teacher_model,
        student_model_path=args.student_model,
        device="cpu"
    )

    # Create training texts
    training_texts = create_training_texts(args.num_samples)
    print(f"Created {len(training_texts)} training samples")

    # Train with distillation
    trainer.train_distillation(
        texts=training_texts,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.output_model
    )

    # Evaluate if requested
    if args.evaluate:
        test_texts = create_training_texts(10)  # Use 10 samples for testing
        results = trainer.evaluate_distillation(test_texts)

    print("\nKnowledge distillation complete!")
    print("SimpleTTS has been trained to mimic NeuTTS behavior.")
    print(f"Distilled model saved to: {args.output_model}")


if __name__ == "__main__":
    main()