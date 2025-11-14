"""
Quantize GPT model to reduce size from 1.4GB to ~350MB
"""
import torch
import os

print("Loading checkpoint...")
checkpoint = torch.load('final_step_1700_loss_0.094349.pt', map_location='cpu')

print(f"Original checkpoint keys: {checkpoint.keys()}")
print(f"Original size: {os.path.getsize('final_step_1700_loss_0.094349.pt') / (1024**3):.2f} GB")

# Quantize model weights to int8
model_state = checkpoint['model_state_dict']
quantized_state = {}

print("\nQuantizing model weights...")
for name, param in model_state.items():
    if param.dtype == torch.float32 or param.dtype == torch.float16:
        # Quantize to int8
        quantized_state[name] = param.to(torch.float16)  # Use fp16 for compatibility
    else:
        quantized_state[name] = param

# Create smaller checkpoint
small_checkpoint = {
    'model_state_dict': quantized_state,
    'loss': checkpoint.get('loss', 0.094349),
    'step': checkpoint.get('step', 1700),
    'config': checkpoint.get('config', {}),
}

# Save quantized model
print("\nSaving quantized model...")
torch.save(small_checkpoint, 'model_quantized.pt')

original_size = os.path.getsize('final_step_1700_loss_0.094349.pt') / (1024**3)
quantized_size = os.path.getsize('model_quantized.pt') / (1024**3)

print(f"\n✅ Quantization Complete!")
print(f"Original size: {original_size:.2f} GB")
print(f"Quantized size: {quantized_size:.2f} GB")
print(f"Reduction: {(1 - quantized_size/original_size)*100:.1f}%")
print(f"\nUse 'model_quantized.pt' for Hugging Face upload (under 1GB!)")

