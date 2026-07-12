# Alary25B

Alary25B is an open-source large language model (LLM) developed from scratch by the AlaryLLM project.

## Project Status

**Current Stage:** 🚧 In Development

## Roadmap

- ✅ Design the model architecture
- ✅ Build the training pipeline
- ✅ Build and train the tokenizer
- ✅ Pre-training data preparation (7T tokens)
- ⏳ Pre-train the base model
- ⏳ Instruction tuning with Aya dataset
- ⏳ Evaluate the model
- ⏳ Release the first public checkpoint
- ⏳ Release AlaryChat

## Features

- 25B parameter Llama-style architecture
- Trained from scratch
- Custom 128K BPE tokenizer
- Long context support (2048 tokens)
- Multilingual support
- Code generation
- Instruction-following capability
- Open-source (Apache 2.0)

## Training Data

### Pre-training (Base Model)
The model is pre-trained using a mixture of:

- **CulturaX** (80%) - Multilingual general text
- **StarCoderData** (10%) - Code data
- **The Stack v2** (10%) - Additional code

**Total:** 7 Trillion tokens

### Instruction Tuning
Fine-tuned on **Aya Dataset** for instruction-following capability

## Repository Structure

```
Alary25B/
├── model.py                 # Creates the model architecture (25B params)
├── tokenizer_trainer.py     # Trains the 128K BPE tokenizer
├── dataset.py               # Loads and tokenizes the pre-training data
├── train.py                 # Pre-training script (TPU optimized)
├── instruction_train.py     # Instruction tuning script (Aya dataset)
├── hf_loader.py             # Uploads the model to Hugging Face Hub
├── README.md                # This file
└── LICENSE                  # Apache 2.0

Outputs/
├── Alary25B/                # Pre-trained model checkpoint
├── Alary25B_Final/          # Final pre-trained model
├── instruction_tuning_checkpoints/  # Instruction tuning checkpoints
└── checkpoints/             # Training checkpoints
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.35+
- Accelerate 0.24+
- Datasets 2.14+
- Tokenizers 0.14+
- Hugging Face Hub
- PEFT (for LoRA fine-tuning)

### Optional (for local training)
- CUDA 11.8+ (for GPU training)
- flash-attn (for faster attention)

## Installation

```bash
# Clone the repository
git clone https://github.com/gokbagbasi-creator/Alary21B.git
cd Alary21B

# Install dependencies
pip install torch transformers accelerate datasets tokenizers huggingface-hub peft

# Optional: For TPU support
pip install cloud-tpu-client
```

## Quick Start

### Stage 1: Model Initialization

Initialize the 25B parameter model architecture:

```bash
python model.py
```

**Output:**
- `Alary25B/` - Model configuration and initial weights

### Stage 2: Tokenizer Training

Train the 128K BPE tokenizer on the mixed datasets:

```bash
python tokenizer_trainer.py
```

**Output:**
- `tokenizer.json` - Trained BPE tokenizer

### Stage 3: Pre-training (Base Model)

Train the model on 7 trillion tokens with TPU optimization:

```bash
# TPU configuration (recommended)
python train.py

# Or with Accelerate for multi-GPU
accelerate launch --multi_gpu train.py
```

**Configuration (train.py):**
- Batch size: 4 (adjust for your hardware)
- Learning rate: 1e-4
- Mixed precision: bfloat16 (TPU native)
- Gradient accumulation: 8
- Max sequence length: 4096

**Output:**
- `checkpoints/` - Training checkpoints (every 1000 steps)
- `Alary25B_Final/` - Final pre-trained model

**TPU Performance:**
- **TPU v4 Pod (8 chips):** ~5-8 days for 7T tokens
- **8x A100 GPU:** ~28 days
- **Single GPU:** Not practical

### Stage 4: Instruction Tuning

Fine-tune the pre-trained model on Aya dataset for instruction-following:

```bash
python instruction_train.py
```

**Configuration (instruction_train.py):**
```python
InstructionTuningConfig(
    model_id="Alary25B_Final",
    dataset_name="aya_dataset",  # or local path
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=5e-5,
    max_seq_length=2048,
)
```

**Optional: Use LoRA for efficient fine-tuning**
```python
config.use_lora = True  # Reduces memory usage by 70%
```

**Output:**
- `instruction_tuning_checkpoints/` - Training checkpoints
- `instruction_tuning_checkpoints/final_model/` - Final instruction-tuned model

### Stage 5: Upload to Hugging Face Hub

Upload the trained model and tokenizer to Hugging Face:

```bash
python hf_loader.py
```

**Configuration (hf_loader.py):**
Edit the `REPO_ID` before running:
```python
REPO_ID = "YourUsername/Alary25B"  # Change this
```

**Output:**
- Model uploaded to Hugging Face Hub
- Publicly accessible for everyone

## Advanced Configuration

### Pre-training Hyperparameters

Edit `train.py` to customize:

```python
# Model architecture
hidden_size=5120
num_hidden_layers=64
num_attention_heads=32
vocab_size=128000
max_position_embeddings=128000

# Training
learning_rate=1e-4
gradient_accumulation_steps=8
batch_size=4
warmup_steps=500
```

### Instruction Tuning Hyperparameters

Edit `instruction_train.py` to customize:

```python
# Training
num_train_epochs=3
per_device_train_batch_size=16
learning_rate=5e-5
max_seq_length=2048

# LoRA (optional)
use_lora=True
lora_r=16
lora_alpha=32
```

### Data Processing

The `dataset.py` and `instruction_train.py` handle:
- Automatic dataset downloading
- Streaming to reduce memory usage
- Tokenization with padding/truncation
- Batch processing

## Training on TPU Research Cloud

### Setup TPU Environment

```bash
# Install TPU dependencies
pip install cloud-tpu-client

# Set TPU credentials
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

### Launch TPU Training

```bash
# TPU v4 Pod (recommended)
python train.py --use_tpu=True

# Monitor training
tensorboard --logdir=./logs
```

### Expected Performance

| Configuration | Time | Cost |
|---------------|------|------|
| TPU v4 (8 chips) | 5-8 days | Free (Research Cloud) |
| TPU v5 (8 chips) | 3-5 days | Free (Research Cloud) |
| 8x A100 | 28 days | ~$10K |
| Single A100 | 3+ months | ~$2K |

## Model Specifications

### Architecture Details

| Parameter | Value |
|-----------|-------|
| **Total Parameters** | 25B |
| **Hidden Size** | 5120 |
| **Intermediate Size** | 20480 |
| **Number of Layers** | 64 |
| **Attention Heads** | 32 |
| **KV Heads** | 8 |
| **Vocab Size** | 128000 |
| **Max Sequence Length** | 128000 |
| **Position Embeddings** | RoPE |

## Model Usage

### Using the Pre-trained Model

```python
from transformers import LlamaForCausalLM, AutoTokenizer

model_id = "YourUsername/Alary25B"
model = LlamaForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Generate text
prompt = "Merhaba, nasılsın?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

### Using the Instruction-Tuned Model

```python
model_id = "YourUsername/Alary25B-Instruct"
model = LlamaForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Instruction-following format
prompt = """### Instruction:
Türkçe'den İngilizce'ye çevir.

### Input:
Merhaba, bu bir test.

### Response:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))
```

## Troubleshooting

### Out of Memory (OOM) Error

```python
# Reduce batch size
batch_size = 2  # Instead of 4

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
mixed_precision = "bf16"
```

### Slow Tokenization

```python
# Use num_workers in DataLoader
train_dl = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,  # Increase this
    pin_memory=True
)
```

### TPU Not Found

```bash
# Check TPU availability
python -c "import jax; print(jax.devices())"

# Reinstall JAX for TPU
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Training Monitoring

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir=./logs

# View at http://localhost:6006
```

### Weights & Biases (Optional)

```python
# Add to train.py
report_to=["wandb"]

# Login
wandb login
```

## Results & Benchmarks

Coming soon after pre-training completion...

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## AI Assistance

This project was developed with the assistance of AI tools. AI was used to help generate, improve, debug, and refine parts of the code. The overall project design, architecture, integration, and testing were done by the project author.

## License

Apache License 2.0 - See LICENSE file for details

## Citation

If you use Alary25B in your research, please cite:

```bibtex
@software{alary25b,
  title={Alary25B: An Open-Source 25B Large Language Model},
  author={Göktürk97},
  year={2026},
  url={https://github.com/gokbagbasi-creator/Alary21B}
}
```

## Author

**Göktürk97** - [GitHub](https://github.com/gokbagbasi-creator)

**Project:** AlaryLLM - Building open-source language models from scratch

## Acknowledgments

- Hugging Face 🤗 for transformers and datasets
- Google TPU Research Cloud for computational resources
- Cohere for Aya dataset
- Meta for Llama architecture

## Support

For issues, questions, or suggestions:

- 📝 Open an issue on GitHub
- 💬 GitHub Discussions
- 📧 Contact: gokbagbasi@gmail.com

---

**Last Updated:** July 12, 2026
