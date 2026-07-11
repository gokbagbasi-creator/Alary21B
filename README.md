Alary21B

Alary21B is an open-source large language model (LLM) developed from scratch by the AlaryLLM project.

Project Status

Current Stage: 🚧 In Development

Roadmap

- ✅ Design the model architecture
- ✅ Build the training pipeline
- ✅ Build and train the tokenizer
- ⏳ Train the base model
- ⏳ Evaluate the model
- ⏳ Release the first public checkpoint
- ⏳ Instruction tuning
- ⏳ Release AlaryChat

Features

- 21B parameter Llama-style architecture
- Trained from scratch
- Custom 128K BPE tokenizer
- Long context support
- Multilingual support
- Code generation
- Open-source

Training Data

The model is trained using a mixture of:

- CulturaX
- StarCoderData
- The Stack v2

Repository Structure

- "model.py" – Creates the model architecture.
- "train_tokenizer.py" – Trains the tokenizer.
- "data_loader.py" – Loads and tokenizes the training data.
- "train.py" – Runs model training.
- "upload_hf.py" – Uploads the model to Hugging Face Hub.

Requirements

- Python 3.12+
- PyTorch
- Transformers
- Accelerate
- Datasets
- Tokenizers
- Hugging Face Hub

AI Assistance

This project was developed with the assistance of AI tools. AI was used to help generate, improve, debug, and refine parts of the code. The overall project design, architecture, integration, testing, and final decisions were made by the project author.

License

Apache License 2.0

Author

Göktürk97

Project: AlaryLLM
