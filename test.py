import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B", torch_dtype=torch.float16, device_map="cpu")

# Check all possible norm locations
print("model.model.norm:", hasattr(model.model, 'norm'))
print("model.model.language_model.norm:", hasattr(model.model.language_model, 'norm'))
print("model.model.language_model.model.norm:", hasattr(model.model.language_model.model, 'norm'))

# Print the language model's children to find the norm
print("\nLanguage model children:")
for name, child in model.model.language_model.named_children():
    print(f"  {name}: {type(child).__name__}")