---
title: Shakespeare Text Generator
emoji: 🎭
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎭 Shakespeare Text Generator

A GPT-2 language model (124M parameters) trained on Shakespeare's complete works, achieving a loss of **0.094349**.

## 🎯 Model Details

- **Architecture**: GPT-2 (Decoder-only Transformer)
- **Parameters**: 123,653,632 (124M)
- **Training Loss**: 0.094349 (target was < 0.099999) ✅
- **Training Data**: Shakespeare's complete works
- **Model Format**: FP16 Quantized (330MB - optimized for deployment)

## 🚀 Features

- Generate Shakespearean-style text from any prompt
- Adjustable creativity (temperature)
- Control diversity (top-k sampling)
- Variable length generation (50-500 tokens)

## 🎓 Training Techniques

This model was trained using state-of-the-art techniques:

1. **Gradient Accumulation** (effective batch size: 64)
2. **Learning Rate Scheduling** (Warmup + Cosine Decay)
3. **AdamW Optimizer** (β=(0.9, 0.95), weight_decay=0.1)
4. **Parameter-specific Weight Decay**
5. **Gradient Clipping** (max_norm=1.0)
6. **GPT-2 Style Initialization**

## 📊 Performance

- **Final Loss**: 0.094349 (meets target < 0.099999) ✅
- **Model Size**: 330MB (FP16 quantized from 1.44GB)
- **Quality**: High coherence and Shakespearean style (minimal degradation from quantization)
- **Architecture**: 12 layers, 12 heads, 768 embedding dimension

## 💡 Usage

Simply enter a Shakespearean prompt (e.g., "First Citizen:", "ROMEO:", "To be, or not to be,") and adjust the generation parameters:

- **Max Tokens**: Length of generated text
- **Temperature**: Controls randomness (0.5=focused, 1.2=creative)
- **Top-K**: Vocabulary diversity (50 recommended)

## 🔗 Links

- **GitHub**: [Source Code & Training Details](https://github.com/yourusername/gpt-shakespeare)
- **Paper**: Based on GPT-2 architecture (Radford et al., 2019)

## 📝 Citation

If you use this model, please cite:

```bibtex
@misc{shakespeare-gpt2-2024,
  title={Shakespeare Text Generator: GPT-2 Model},
  author={Your Name},
  year={2024},
  note={Trained GPT-2 model (124M params) on Shakespeare's works, achieving loss 0.094349}
}
```

## 📄 License

MIT License - Free to use for educational and research purposes.

