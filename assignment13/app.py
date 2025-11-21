import gradio as gr
import torch
import os
from transformers import AutoTokenizer
from model import Llama, LlamaConfig

# Load model
device = 'cpu' # Spaces usually use CPU unless GPU upgrade
config = LlamaConfig()
model = Llama(config)

# Load weights (assuming final_model.pt is uploaded)
if os.path.exists("final_model.pt"):
    state_dict = torch.load("final_model.pt", map_location=device)
    model.load_state_dict(state_dict)
else:
    print("No model found, using random weights")

model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

def generate_text(prompt, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            
    return tokenizer.decode(input_ids[0].tolist())

iface = gr.Interface(
    fn=generate_text,
    inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here...", label="Input Prompt"),
    outputs=gr.Textbox(lines=15, label="Generated Output", show_copy_button=True),
    title="SmolLM2-135M Demo",
    description="Reverse engineered and trained SmolLM2-135M model (Shakespearean style).",
    examples=[
        ["The king hath sent for"],
        ["My lord, the army is"],
        ["What say you to this"]
    ]
)

if __name__ == "__main__":
    iface.launch()
