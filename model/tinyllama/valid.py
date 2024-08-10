import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os 
from pathlib import Path
from tqdm.auto import tqdm

model_id = os.getcwd()
if len(sys.argv) == 2:
    filename = sys.argv[1]
elif len(sys.argv) == 3:
    filename = sys.argv[1]
    model_id = sys.argv[2]
else:
    raise Exception("use valid.py <path-to-text> [model-id]")

text = Path(filename).read_text()
stories = text.split("<|endoftext|>")
print(len(stories))
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).cuda().bfloat16()

ctx_size = tokenizer.model_max_length
sliding_window = ctx_size // 2

total_loss = 0.0
measurements = 0
model.eval()
for story in (bar := tqdm(stories)):
    story = story.strip()
    tokens = tokenizer(story, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    i = 0
    while i < len(tokens):
        current_window = tokens[i:i+ctx_size-1]
        part_tokens = [tokenizer.bos_token_id] + current_window
        input_ids = torch.tensor(part_tokens, device="cuda")[None]
        labels = input_ids.clone()
        if i:
            # disable seen tokens
            labels[:, :-sliding_window] = -100

        with torch.no_grad():
            loss = model(input_ids, labels=labels).loss
            total_loss += loss.item() 
            measurements += 1

        i += len(current_window)
        bar.set_description(f"L {total_loss/measurements:.4f}")

print(f"FINAL LOSS: {total_loss/measurements:.4f}")


