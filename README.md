# TTS Model Project

This project implements a Text-to-Speech (TTS) model with various components including distillation, training, and inference scripts.

## Project Structure

- `src/`: Core TTS inference scripts
  - `rag_tts.py`: RAG-based TTS implementation
  - `tts.py`: Additional TTS utilities
- `data/`: Dataset files (ignored in Git)
- `models/`: Model files (ignored in Git)
- `setup.md`: Setup instructions
- `training/`: Training scripts (work in progress, ignored in Git)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd tts-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Note: Ensure you have Python 3.x and necessary libraries like torch, etc.)

## Usage

### Running TTS Inference

**RAG-based TTS:**
```bash
python src/rag_tts.py --text "Hello, world!" --output output.wav
```

**Basic TTS:**
```bash
python src/tts.py --text "Hello, world!" --output output.wav
```

### Command Line Options

Both scripts support the following options:
- `--text`: Text to convert to speech
- `--output`: Output audio file path (default: output.wav)
- `--model`: Path to pre-trained model (if available)
- `--speaker`: Speaker ID for multi-speaker models (default: 0)

### Example Usage

```bash
# Generate speech with RAG TTS
python src/rag_tts.py --text "This is a test of the RAG-based TTS system." --output rag_test.wav

# Generate speech with basic TTS
python src/tts.py --text "Hello, how are you today?" --output basic_test.wav
```

Refer to `setup.md` for detailed setup instructions and model requirements.

## Contributing

Feel free to contribute by opening issues or pull requests.

## License

[Add license if any]