# Alary21B

Alary21B is an open-source large language model (LLM) developed from scratch by the AlaryLLM project.

## Project Status

**Current Stage:** 🚧 In Development

## Roadmap

- ✅ Design the model architecture
- ✅ Build the training pipeline
- ✅ Build and train the tokenizer
- ⏳ Train the base model
- ⏳ Evaluate the model
- ⏳ Release the first public checkpoint
- ⏳ Instruction tuning
- ⏳ Release AlaryChat

## Features

- 21B parameter Llama-style architecture
- Trained from scratch
- Custom 128K BPE tokenizer
- Long context support
- Multilingual support
- Code generation
- Open-source

## Training Data

The model is trained using a mixture of:

- CulturaX
- StarCoderData
- The Stack v2

## Repository Structure

```
├── model.py                 # Creates the model architecture
├── tokenizer_trainer.py     # Trains the tokenizer
├── dataset.py               # Loads and tokenizes the training data
├── train.py                 # Runs model training
├── hf_loader.py             # Uploads the model to Hugging Face Hub
└── README.md                # This file
```

## Requirements

- Python 3.12+
- PyTorch
- Transformers
- Accelerate
- Datasets
- Tokenizers
- Hugging Face Hub

## Installation

```bash
pip install torch transformers accelerate datasets tokenizers huggingface-hub
```

## Quick Start

### 1. Create Model Architecture

```bash
python model.py
```

This creates the Llama-style model configuration and saves it to the `Alary21B` directory.

### 2. Train Tokenizer

```bash
python tokenizer_trainer.py
```

This trains a 128K BPE tokenizer on the mixed datasets and saves it as `tokenizer.json`.

### 3. Train Model

```bash
python train.py
```

This starts the model training with mixed precision (bf16) and gradient accumulation.

### 4. Upload to Hugging Face Hub

```bash
python hf_loader.py
```

This uploads the trained model and tokenizer to Hugging Face Hub.

## Configuration

Edit `model.py` to change model hyperparameters:
- `hidden_size`: 5120 (hidden dimension)
- `num_hidden_layers`: 64 (number of transformer layers)
- `num_attention_heads`: 32 (attention heads)
- `vocab_size`: 128000 (tokenizer vocabulary size)
- `max_position_embeddings`: 128000 (context length)

## Training Configuration

Edit `train.py` to adjust:
- `batch_size`: Training batch size (default: 4)
- `gradient_accumulation_steps`: Accumulation steps (default: 8)
- `learning_rate`: Learning rate (default: 1e-4)

## Resuming Training

Training automatically resumes from the latest checkpoint in the `checkpoints` directory.

## AI Assistance

This project was developed with the assistance of AI tools. AI was used to help generate, improve, debug, and refine parts of the code. The overall project design, architecture, integration, and testing were done by the project author.

## License

Apache License 2.0

## Author

Göktürk97

**Project:** AlaryLLM
