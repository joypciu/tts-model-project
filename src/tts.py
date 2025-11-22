#!/usr/bin/env python3
"""
bench_neutts.py
Simple benchmarking script for NeuTTS Air (GGUF / local file or HF repo).
Saves output wav and prints seconds-per-second (wall_time / generated_seconds).

Usage examples:
  # set threads (tune for your CPU; i7-12700 -> try 12 or 20)
  OMP_NUM_THREADS=20 MKL_NUM_THREADS=20 python bench_neutts.py \
      --backbone /path/to/neutts-air-q4-gguf.gguf \
      --codec neuphonic/neucodec \
      --ref_audio samples/dave.wav \
      --ref_text samples/dave.txt \
      --text "Hello, this is a latency test."

  # or using HF repo id for backbone (if you prefer)
  python bench_neutts.py --backbone neuphonic/neutts-air-q4-gguf \
      --codec neuphonic/neucodec --ref_audio samples/dave.wav \
      --ref_text samples/dave.txt --text "hello world"
"""

import argparse
import time
import soundfile as sf
import os
import sys

# Set espeak environment variables for phonemizer
os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = r'C:\Program Files\eSpeak NG\libespeak-ng.dll'
os.environ['PHONEMIZER_ESPEAK_PATH'] = r'C:\Program Files\eSpeak NG'
import warnings
# Suppress TensorFlow warnings for cleaner output
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')


# Add the local neutts-air directory to the path for importing neuttsair
sys.path.insert(0, 'neutts-air')

# Try to import the library wrapper used by the repo.
# The README examples show `from neuttsair.neutts import NeuTTSAir`
try:
    from neuttsair.neutts import NeuTTSAir
except Exception as e:
    print("Failed to import neuttsair.neutts. Make sure you installed requirements:")
    print("  pip install -r requirements.txt")
    print("and that you are running this script from a place where the installed package is available.")
    print("Import error:", e)
    sys.exit(1)


def read_ref_text(path):
    if not path:
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    p = argparse.ArgumentParser(description="Bench NeuTTS Air (GGUF) - measure core inference latency")
    p.add_argument("--backbone", default="neutss-air-BF16.gguf",
                   help="Backbone GGUF file path or HF repo id (e.g. /path/to/neutts-air-q4-gguf.gguf or neuphonic/neutts-air-q4-gguf)")
    p.add_argument("--backbone_device", default="cpu", choices=["cpu", "gpu"],
                   help="Device for the backbone. Use 'cpu' for CPU/GGML. (Repo examples show CPU default.)")
    p.add_argument("--codec", default="neuphonic/neucodec",
                   help="Codec repo id or local path. (Default: neuphonic/neucodec)")
    p.add_argument("--codec_device", default="cpu", choices=["cpu", "gpu"],
                   help="Device for codec.")
    p.add_argument("--ref_audio", default="neutts-air/samples/dave.wav", help="Reference WAV (mono) for voice cloning.")
    p.add_argument("--ref_text", default="neutts-air/samples/dave.txt", help="Text that matches the reference audio (plain text file).")
    p.add_argument("--text", help="Text to synthesize.")
    p.add_argument("--out", default="out.wav", help="Output WAV filename (24 kHz recommended by repo).")
    p.add_argument("--sample_rate", type=int, default=24000, help="Output sample rate (default 24000).")
    p.add_argument("--no_codec", action="store_true", help="Skip codec encode/decode measurement (if you only want LM infer).")
    p.add_argument("--verbose", action="store_true", help="Verbose prints.")
    args = p.parse_args()

    # Prompt for text if not provided
    if not args.text:
        args.text = input("Enter the sentence to synthesize: ")

    # Print environment hints (helpful when debugging perf)
    if args.verbose:
        print("Environment variables (relevant):")
        for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            print(f"  {k} = {os.environ.get(k)}")
        print("backbone:", args.backbone)
        print("codec:", args.codec)
        print("backbone_device:", args.backbone_device, "codec_device:", args.codec_device)

    # Load reference text
    ref_text = read_ref_text(args.ref_text)
    if args.verbose:
        print("Reference text length:", len(ref_text))

    # Instantiate TTS
    # The README shows usage like: NeuTTSAir(backbone_repo="neuphonic/neutts-air-q4-gguf", backbone_device="cpu", codec_repo="neuphonic/neucodec")
    tts = NeuTTSAir(
        backbone_repo=args.backbone,
        backbone_device=args.backbone_device,
        codec_repo=args.codec,
        codec_device=args.codec_device
    )

    # Encode reference audio -> codes (this will call the codec.encode_reference)
    if args.verbose:
        print("Encoding reference audio:", args.ref_audio)
    start = time.perf_counter()
    ref_codes = tts.encode_reference(args.ref_audio)
    encode_time = time.perf_counter() - start
    if args.verbose:
        print(f"Reference encoding time: {encode_time:.3f} s")

    # Now measure the core TTS inference (LM + generator).
    # The README/example uses: wav = tts.infer(input_text, ref_codes, ref_text)
    # We'll measure only the infer() call (not file IO).
    if args.verbose:
        print("Starting core inference (tts.infer)...")
    start_infer = time.perf_counter()
    wav = tts.infer(args.text, ref_codes, ref_text)
    infer_time = time.perf_counter() - start_infer

    # wav is expected to be a numpy array at sample_rate (repo examples show 24000)
    # Compute generated audio length in seconds
    if hasattr(wav, "shape"):
        length_seconds = float(len(wav)) / float(args.sample_rate)
    else:
        # if it's e.g. a list
        length_seconds = float(len(wav)) / float(args.sample_rate)

    # Save WAV
    if args.verbose:
        print(f"Saving output to {args.out} (sr={args.sample_rate})")
    sf.write(args.out, wav, args.sample_rate)

    # Print results
    print("=== NeuTTS Air benchmark results ===")
    print(f"Reference encoding time: {encode_time:.3f} s")
    print(f"Core inference time (wall): {infer_time:.3f} s")
    print(f"Generated audio length: {length_seconds:.3f} s")
    if length_seconds > 0:
        sec_per_sec = infer_time / length_seconds
        print(f"Seconds of wall time per 1 second of audio: {sec_per_sec:.3f} s/sec")
        print(f"Real-time factor (RTF) = 1 / (sec_per_sec). >1 means faster-than-real-time: {1.0/sec_per_sec:.2f}x")
    else:
        print("Warning: generated audio length is zero or could not be computed.")

    print("Output file:", os.path.abspath(args.out))
    print("If you want to measure only LM time, set --no_codec (but the API above measured codec separated).")

if __name__ == "__main__":
    main()
