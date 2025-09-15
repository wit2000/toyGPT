# Self-Trained Toy GPT

Experimental: This self-trained model is primarily for learning purposes. It currently cannot fully understand language or converse like a typical chatbot, but it can generate output based on the learned words.

## Features
- Self-trainable GPT model (word-level or character-level)
- Lightweight dataset support (e.g., tiny Shakespeare or light novels)
- Interactive chat interface after training
- Customizable hyperparameters

## Requirements
Windows 10/11 or Linux
Python 3.10+
At least 16GB GPU memory recommended

Hardware & CUDA Notes

GPU Used: NVIDIA GeForce RTX 5070ti (16GB VRAM)

CUDA Toolkit Installed: CUDA 12.8 (PyTorch build with cu128) as it support my GPU, refer to official notes to check which CUDA Toolkit is supported by your GPU.

# Environment Setup

Create a Conda environment:
```bash
conda create -n llm python=3.10
conda activate llm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

Place the .txt file in the same directory as your training script (train_toy_gpt.py).

Run Model: python train_toy_gpt.py

```

# Interactive Chat

After training, the model will be saved as model_word.pth.

You can chat with it using your interactive script:

python chat_toy_gpt_word.py


Word-level models generate output based on learned words.

Performance improves with larger models, more iterations, and more context.

Notes from Experiments

Dataset sizes tested: 245k–1M characters; larger datasets improve contextual understanding.

Character-level models produce more coherent English structure but converge slower.

Word-level models require building vocab from words, not characters.

Output quality improves with:

Larger model parameters

More training iterations

Larger context (block_size)

# Pre-trained (Hugging face model)

# Mistral-7B Local Setup

This repository contains instructions and scripts to set up and run the Mistral-7B-Instruct model locally on Windows with GPU support (CUDA).

## Features
- Load Mistral-7B-Instruct-v0.3
- Simple chat interface
- Optional context memory
- GPU support

## Requirements
- Windows 10/11 or Linux
- Python 3.10+
- At least 16GB GPU memory recommended
- ~40GB free disk space

# Mistral-7B Setup Instructions

## 1. Create Python Environment
```bash
conda create -n llm python=3.10
conda activate llm

pip install -r requirements.txt
huggingface-cli login

from transformers import AutoModelForCausalLM, AutoTokenizer

Notes: Ensure torch.cuda.is_available() returns True. Else it will use CPU and won't be fast to response.

Run Model:
python scripts/test_model.py
```




