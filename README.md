# TTS Model Project

This project implements a Text-to-Speech (TTS) model with various components including distillation, training, and inference scripts.

## Project Structure

- `src/`: Core TTS inference scripts
  - `rag_tts.py`: RAG-based TTS
  - `tts.py`: Additional TTS utilities
- `training/`: Training and distillation scripts
  - `simple_tts.py`: Simple TTS implementation with training
  - `train_vocoder.py`: Vocoder training script
  - `distill_train.py`: Distillation training script
  - `distill_finetune.py`: Distillation fine-tuning script
  - `generate_dataset.py`: Dataset generation script
- `data/`: Dataset files (ignored in Git)
- `models/`: Model files (ignored in Git)
- `setup.md`: Setup instructions

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

- Run simple TTS: `python src/simple_tts.py`
- Generate dataset: `python src/generate_dataset.py`
- Train model: `python src/distill_train.py`
- Fine-tune: `python src/distill_finetune.py`

Refer to `setup.md` for detailed setup instructions.

## Contributing

Feel free to contribute by opening issues or pull requests.

## License

[Add license if any]