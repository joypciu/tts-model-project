#!/usr/bin/env python3
"""
Vocoder Training Script
Trains the WaveNet vocoder in SimpleTTS on the Toronto Emotional Speech Set.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm
import librosa
import soundfile as sf
from simple_tts import SimpleTTS
from typing import Optional


class VocoderDataset(Dataset):
    """Dataset for vocoder training using TESS dataset."""

    def __init__(self, data_root: str, sample_rate: int = 22050, max_length: int = 50000):
        self.sample_rate = sample_rate
        self.max_length = max_length

        # Find all WAV files in the dataset
        self.wav_files = []
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.wav'):
                    self.wav_files.append(os.path.join(root, file))

        print(f"Found {len(self.wav_files)} WAV files for vocoder training")

        if len(self.wav_files) == 0:
            raise ValueError(f"No WAV files found in {data_root}")

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]

        try:
            # Load audio
            audio, sr = librosa.load(wav_path, sr=self.sample_rate, mono=True)

            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            # Pad or truncate to fixed length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, self.max_length - len(audio)), 'constant')

            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)

            # Extract mel spectrogram as conditioning input
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=1024,
                hop_length=256,
                n_mels=80,
                fmin=0,
                fmax=8000
            )

            # Convert to log scale and normalize
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

            # Convert to tensor
            mel_tensor = torch.tensor(mel_spec, dtype=torch.float32)

            return {
                'mel_spec': mel_tensor,
                'audio': audio_tensor,
                'audio_length': len(audio)  # Note: this is the original length before padding
            }

        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            # Return dummy data
            return {
                'mel_spec': torch.zeros(80, 196),  # 80 mel bins, ~196 time steps for 50k samples
                'audio': torch.zeros(self.max_length),
                'audio_length': self.max_length
            }


class VocoderTrainer:
    """Trainer for the vocoder component."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

        # Create a SimpleTTS instance just to access the vocoder
        self.tts = SimpleTTS(device=device)

        # Extract vocoder from TTS model
        self.vocoder = self.tts.model.vocoder

        print(f"Vocoder has {sum(p.numel() for p in self.vocoder.parameters())} parameters")

    def train(self, data_root: str, epochs: int = 10, batch_size: int = 8,
              learning_rate: float = 1e-4, save_path: str = "./trained_vocoder.pth"):
        """Train the vocoder on the speech dataset."""

        # Create dataset and dataloader
        dataset = VocoderDataset(data_root)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.vocoder.parameters(), lr=learning_rate)
        mse_loss = nn.MSELoss()

        self.vocoder.train()

        print(f"Starting vocoder training for {epochs} epochs on {len(dataset)} samples...")

        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                mel_specs = batch['mel_spec'].to(self.device)
                target_audio = batch['audio'].to(self.device)

                optimizer.zero_grad()

                # Generate audio from mel spectrogram
                pred_audio = self.vocoder(mel_specs, speaker_id=0)

                # Ensure shapes match for loss computation
                # pred_audio shape: [batch, 1, time]
                # target_audio shape: [batch, time]
                if pred_audio.dim() == 3:
                    pred_audio = pred_audio.squeeze(1)  # Remove channel dimension

                # Make sure they have the same length
                min_length = min(pred_audio.shape[1], target_audio.shape[1])
                pred_audio = pred_audio[:, :min_length]
                target_audio = target_audio[:, :min_length]

                # Compute loss
                loss = mse_loss(pred_audio, target_audio)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            print(".6f")

        # Save trained vocoder
        torch.save(self.vocoder.state_dict(), save_path)
        print(f"Vocoder training completed! Model saved to {save_path}")

        return save_path

    def test_vocoder(self, test_audio_path: Optional[str] = None):
        """Test the trained vocoder on a sample."""
        self.vocoder.eval()

        if test_audio_path and os.path.exists(test_audio_path):
            # Load test audio
            audio, sr = librosa.load(test_audio_path, sr=22050, mono=True)
            audio = audio / np.max(np.abs(audio))

            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=22050,
                n_fft=1024,
                hop_length=256,
                n_mels=80,
                fmin=0,
                fmax=8000
            )
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

            mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Generate audio
            with torch.no_grad():
                generated = self.vocoder(mel_tensor, speaker_id=0)
                generated = generated.cpu().numpy().squeeze()

            # Save comparison
            sf.write("original_test.wav", audio, 22050)
            sf.write("vocoder_reconstruction.wav", generated, 22050)

            print("Test audio saved: original_test.wav, vocoder_reconstruction.wav")
        else:
            print("No test audio provided or file not found")


def main():
    parser = argparse.ArgumentParser(description="Train vocoder on Toronto Emotional Speech Set")
    parser.add_argument("--data_root", default="./toronto_speech_data/TESS Toronto emotional speech set data",
                       help="Path to TESS dataset root")
    parser.add_argument("--output_model", default="./trained_vocoder.pth",
                       help="Path to save trained vocoder")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--test_audio", default=None,
                       help="Path to test audio file for reconstruction test")

    args = parser.parse_args()

    # Create vocoder trainer
    trainer = VocoderTrainer(device="cpu")

    # Train vocoder
    trainer.train(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_path=args.output_model
    )

    # Test vocoder if test audio provided
    if args.test_audio:
        trainer.test_vocoder(args.test_audio)

    print(f"\nVocoder training complete! Model saved to {args.output_model}")
    print("You can now use this trained vocoder for better TTS quality.")


if __name__ == "__main__":
    main()