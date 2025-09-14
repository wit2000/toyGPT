1. Environment Setup

Create a Conda environment:

conda create -n llm python=3.11
conda activate llm


Install required packages:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


Optional: uninstall previous conflicting packages:

pip uninstall transformers accelerate safetensors -y

2. Prepare Dataset

Download dataset (example with Shakespeare):

Notes: I used my own dataset copy from a light novel

wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O shakespeare.txt


Place the .txt file in the same directory as your training script (train_toy_gpt.py).

Ensure file extension is correct (shakespeare.txt, not shakespeare.txt.txt).

3. Create/Modify Training Script

Basic dataset loading:

with open("shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()


For word-level tokenization:

words = text.split()  # splits by whitespace
vocab = sorted(set(words))
stoi = {w:i for i,w in enumerate(vocab)}
itos = {i:w for i,w in stoi.items()}
vocab_size = len(vocab)


For char-level tokenization, similar but use chars = sorted(set(text)).

4. Hyperparameters
block_size = 256     # context window
batch_size = 64
n_embd = 512         # embedding size
n_head = 8
n_layer = 6
max_iters = 2800     # adjust as needed
eval_interval = 200

5. Training

Run training script:

python train_toy_gpt.py


Observe train and validation loss every eval_interval steps.

Adjust max_iters and hyperparameters to avoid overfitting.

6. Notes from Experiments

Dataset size: 245k–1M characters tested; larger dataset improves contextual understanding.

Character-level model generates more coherent English structure but slower convergence.

7. Run the training:

conda activate llm   # or your Python environment
python train_toy_gpt.py


After training, the model will be saved as model_word.pth.

Use your interactive script (chat_toy_gpt_word.py) to chat with the model:

python chat_toy_gpt_word.py

Word-level model requires building vocab from words, not characters.

Output improves with larger model parameters, more iterations, and larger context (block_size).
