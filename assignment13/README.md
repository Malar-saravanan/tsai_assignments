# SmolLM2-135M Reverse Engineering & Training

## 🎯 Target
The objective of this project was to reverse engineer the **SmolLM2-135M** model and train it from scratch to verify architectural understanding.
*   **Reverse Engineering**: Reconstruct the exact model architecture (Llama-based) without using the `transformers` library for the model definition.
*   **Training Constraint**: Train for exactly **5000 steps**, stop, save a checkpoint, and then **resume for 50 more steps** to demonstrate state persistence.
*   **Optimization**: Implement speedups like Flash Attention and Mixed Precision.

## 📊 Results
*   **Final Loss**: Decreased from **5.85** (initial) to **0.08** (step 5050).
*   **Convergence**: The model successfully converged on the Shakespeare dataset, demonstrating clear overfitting (expected for this model size on a small dataset) which confirms the capacity of the 135M parameter architecture.
*   **Resumption**: Successfully verified checkpoint loading at step 5000 and continued training to step 5050 without loss spikes.
*   **Generation**: The model generates coherent Shakespearean-style text (e.g., *"Nor neighbour no envy, Nor self-same sorrow..."*).

## 🔍 Analysis
*   **Architecture Choice**: SmolLM2 uses the **Llama** architecture rather than GPT-2. This provides modern stability features like **RMSNorm** and **SwiGLU**.
*   **Efficiency**: The use of **Grouped Query Attention (GQA)** (9 query heads sharing 3 KV heads) significantly reduces memory bandwidth during inference, making this 135M model highly efficient for edge devices.
*   **Positional Embeddings**: **RoPE** allows the model to generalize better to sequence lengths different from training, although we trained on a fixed block size of 512.
*   **Training Dynamics**: The rapid drop in loss indicates the model architecture is highly effective. The use of **Mixed Precision (AMP)** and **Flash Attention** kept training time low on consumer hardware.

---

## 🏗️ Model Architecture
The model is a Transformer decoder based on the Llama architecture.

### Configuration
*   **Hidden Size**: 576
*   **Intermediate Size**: 1536
*   **Layers**: 30
*   **Attention Heads**: 9
*   **KV Heads**: 3 (GQA)
*   **Vocab Size**: 49152

### Parameter Calculation
| Component | Formula | Calculation | Count |
| :--- | :--- | :--- | :--- |
| **Embeddings** | `vocab_size * n_embd` | 49,152 * 576 | 28,311,552 |
| **Attention** (per layer) | | | |
| `q_proj` | `n_embd * n_embd` | 576 * 576 | 331,776 |
| `k_proj` | `n_embd * (n_kv_head * head_dim)` | 576 * (3 * 64) | 110,592 |
| `v_proj` | `n_embd * (n_kv_head * head_dim)` | 576 * 192 | 110,592 |
| `o_proj` | `n_embd * n_embd` | 576 * 576 | 331,776 |
| **MLP** (per layer) | | | |
| `gate_proj` | `n_embd * intermediate_size` | 576 * 1,536 | 884,736 |
| `up_proj` | `n_embd * intermediate_size` | 576 * 1,536 | 884,736 |
| `down_proj` | `intermediate_size * n_embd` | 1,536 * 576 | 884,736 |
| **Norms** (per layer) | `2 * n_embd` | 2 * 576 | 1,152 |
| **Layer Total** | | | **3,540,096** |
| **Total Layers** | `30 * Layer Total` | 30 * 3,540,096 | 106,202,880 |
| **Final Norm** | `n_embd` | 576 | 576 |
| **LM Head** | (Tied with Embeddings) | 0 | 0 |
| **Total** | | | **134,515,008** |

**Approximate Parameter Count**: 135M

## ⚙️ Training Configuration
*   **Dataset**: `input-1.txt` (Shakespeare)
*   **Steps**: 5000 + 50 (Resume)
*   **Batch Size**: 4
*   **Block Size**: 512
*   **Optimizer**: AdamW (LR: 3e-4)
*   **Speedups**:
    *   **Flash Attention**: `F.scaled_dot_product_attention`
    *   **Mixed Precision**: `torch.amp` (bfloat16)
    *   **Pinned Memory**: Data loading optimization

## ✅ Resumption Verification
The training process demonstrates the requirement to stop at step 5000 and resume for 50 more steps.
1.  **Checkpointing**: At step 5000, the model and optimizer states are saved to `checkpoint_5000.pt`.
2.  **Resumption**: The script reloads this checkpoint and continues training.
3.  **Proof**: The logs below show the message `--- Resuming Training ---` followed by `Resumed from step 5000`.

## 📜 Training Log
```
Starting training for 5000 steps...
step 0: loss 5.8471, time 0.46s
Generated: <|endoftext|>Saissosteus in Santagoricinus sinking Tyrion was discovered in Norway, it began to approach - island of Santcha fish 
...
step 500: loss 1.9685, time 4.05s
Generated: <|endoftext|> Nurse? well, 'tis plain he hath done't: look up!  APTISTA: You might have heard him
...
step 1000: loss 0.8429, time 3.85s
...
step 1500: loss 0.4279, time 4.23s
Generated: <|endoftext|>Haste and speed is dear.  NORTHUMBERLAND: And so it is. Forcing him to sleep is 
...
step 2500: loss 0.1854, time 3.75s
Generated: <|endoftext|>I have seen the day of wrong.  Third Citizen: If you will then, he shall have it.
...
step 3000: loss 0.1273, time 3.68s
Generated: <|endoftext|> not another hazard! tumble down, Bear them to our confines. God keep the knight! And, as for ye, against all
...
step 3470: loss 0.1616, time 4.24s
...
step 4500: loss 0.1155, time 3.91s
Generated: <|endoftext|>DUKE VINCENTIO: I know him well: you two with fourscore leave him for Venice; and, after another
...
step 5000: loss 0.0776, time 4.21s
Generated: <|endoftext|>Nor neighbour no envy, Nor self-same sorrow that burns with all eyes Did not let Edward live. Why I am gone, Like
Saving checkpoint_5000.pt...
Checkpoint saved.

--- Resuming Training ---
Resumed from step 5000
Training for 50 more steps...
step 5001: loss 0.0870
step 5002: loss 0.0869
step 5003: loss 0.1026
...
step 5050: loss 0.0820

Training complete.
Final model saved to final_model.pt
```

## 🚀 Deployment
### GitHub Repository
[Link to GitHub Repository](https://github.com/username/repo-name)

### Hugging Face Space
[Link to Hugging Face Space](https://huggingface.co/spaces/malarsaravanan/SmolLM2-135M-demo)
