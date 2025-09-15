# mistral_chat.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1️ Model & tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       # automatically map to GPU
    torch_dtype=torch.float16
)
model.eval()

print("Mistral Chat ready! Type 'quit' to exit.")

# 2️ Chat loop with context
history = []

while True:
    prompt = input("You: ")
    if prompt.lower() == "quit":
        break

    # Append prompt to history
    history.append(prompt)
    # Join full context for model input
    context_text = "\n".join(history)

    inputs = tokenizer(context_text, return_tensors="pt").to(model.device)

    # Generate response
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract only the new part of response
    new_response = response[len(context_text):].strip()
    print("GPT:", new_response)

    # Append model's response to history
    history.append(new_response)
