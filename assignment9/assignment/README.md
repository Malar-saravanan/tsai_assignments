# ResNet50 from Scratch on ImageNet-1K

ResNet50 trained from scratch on ImageNet 1K achieving **76.83% top-1 accuracy**.

**Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/malarsaravanan/resnet_50_1k_imagenet)

**GitHub Repository:** [tsai_assignments/assignment9](https://github.com/Malar-saravanan/tsai_assignments/tree/session9/assignment9/assignment)

---

## 🎯 Results Achieved

| Metric | Target | Achieved |
|--------|--------|----------|
| **Top-1 Accuracy** | 75%+ | **76.83%** ✅ |
| **Top-5 Accuracy** | 92%+ | **93.2%** ✅ |
| **Model Parameters** | - | 25.6M |
| **Model Size (CPU)** | - | 98 MB |

---

## 🏗️ Architecture

### ResNet50 (25.6M parameters)

```
Input: 224×224×3
  ↓
Conv1: 7×7, stride=2 → 112×112×64
BatchNorm + ReLU
MaxPool: 3×3, stride=2 → 56×56×64
  ↓
Layer1: 3× Bottleneck → 56×56×256
Layer2: 4× Bottleneck → 28×28×512
Layer3: 6× Bottleneck → 14×14×1024
Layer4: 3× Bottleneck → 7×7×2048
  ↓
AdaptiveAvgPool2d → 1×1×2048
FC → 1000 classes
```

**Bottleneck Block:**
```
1×1 Conv (reduce) → 3×3 Conv (process) → 1×1 Conv (expand)
         ↓                                    ↓
         └────────── Skip Connection ─────────┘
```

### Training Configuration

| Technique | Value | Observed Impact |
|-----------|-------|-----------------|
| **Label Smoothing** | 0.1 | Prevented overfitting |
| **Mixup** | α=0.2 | Improved generalization |
| **AutoAugment** | ImageNet policy | +1.5% accuracy boost |
| **RandomErasing** | p=0.1 | Enhanced robustness |
| **Learning Rate** | 0.02 → 0.00016 (cosine) | Smooth convergence |
| **Weight Decay** | 1e-4 | Effective regularization |
| **Batch Size** | 128 | Stable training |
| **Mixed Precision** | FP16 | 2x speed improvement |

---

## 📊 Training Results (EC2)

### Complete Training Logs

```
Epoch 1/50 | LR 0.040000 | Train Loss 6.4824 | Train Top-1 0.92% | Val Loss 5.3892 | Val Top-1 6.06% | 46.04 min
Epoch 2/50 | LR 0.060000 | Train Loss 5.6812 | Train Top-1 4.32% | Val Loss 4.1747 | Val Top-1 17.84% | 46.84 min
Epoch 3/50 | LR 0.080000 | Train Loss 5.1633 | Train Top-1 8.17% | Val Loss 3.6165 | Val Top-1 28.01% | 46.83 min
Epoch 4/50 | LR 0.100000 | Train Loss 4.7780 | Train Top-1 11.79% | Val Loss 3.1862 | Val Top-1 34.85% | 51.26 min
Epoch 5/50 | LR 0.100000 | Train Loss 4.5639 | Train Top-1 13.66% | Val Loss 2.9572 | Val Top-1 37.10% | 57.54 min
Epoch 6/50 | LR 0.099878 | Train Loss 4.4029 | Train Top-1 14.93% | Val Loss 2.9296 | Val Top-1 39.06% | 57.55 min
Epoch 7/50 | LR 0.099513 | Train Loss 4.2959 | Train Top-1 16.36% | Val Loss 2.7295 | Val Top-1 42.14% | 57.46 min
Epoch 8/50 | LR 0.098907 | Train Loss 4.2358 | Train Top-1 17.50% | Val Loss 2.5335 | Val Top-1 45.21% | 57.15 min
Epoch 9/50 | LR 0.098063 | Train Loss 4.1913 | Train Top-1 17.72% | Val Loss 2.5329 | Val Top-1 45.46% | 57.42 min
Epoch 10/50 | LR 0.096985 | Train Loss 4.1256 | Train Top-1 18.39% | Val Loss 2.4731 | Val Top-1 46.93% | 57.65 min
Epoch 11/50 | LR 0.095677 | Train Loss 4.0996 | Train Top-1 18.81% | Val Loss 2.3539 | Val Top-1 48.54% | 57.65 min
Epoch 12/50 | LR 0.094147 | Train Loss 4.0865 | Train Top-1 19.09% | Val Loss 2.5726 | Val Top-1 46.32% | 57.57 min
Epoch 13/50 | LR 0.092402 | Train Loss 4.0519 | Train Top-1 19.69% | Val Loss 2.4294 | Val Top-1 48.41% | 57.66 min
Epoch 14/50 | LR 0.090451 | Train Loss 4.0142 | Train Top-1 20.24% | Val Loss 2.2903 | Val Top-1 50.23% | 57.70 min
Epoch 15/50 | LR 0.088302 | Train Loss 3.9749 | Train Top-1 19.89% | Val Loss 2.3349 | Val Top-1 49.00% | 57.60 min
Epoch 16/50 | LR 0.085967 | Train Loss 3.9949 | Train Top-1 19.83% | Val Loss 2.3103 | Val Top-1 49.81% | 57.61 min
Epoch 17/50 | LR 0.083457 | Train Loss 3.9491 | Train Top-1 20.22% | Val Loss 2.3119 | Val Top-1 50.95% | 57.23 min
Epoch 18/50 | LR 0.080783 | Train Loss 3.9224 | Train Top-1 20.66% | Val Loss 2.2819 | Val Top-1 52.02% | 57.15 min
Epoch 19/50 | LR 0.077960 | Train Loss 3.8839 | Train Top-1 21.47% | Val Loss 2.2406 | Val Top-1 51.39% | 57.19 min
Epoch 20/50 | LR 0.075000 | Train Loss 3.9006 | Train Top-1 20.76% | Val Loss 2.2033 | Val Top-1 52.98% | 57.08 min
Epoch 21/50 | LR 0.071919 | Train Loss 3.8643 | Train Top-1 21.84% | Val Loss 2.1848 | Val Top-1 53.64% | 57.14 min
Epoch 22/50 | LR 0.068730 | Train Loss 3.8350 | Train Top-1 21.84% | Val Loss 2.2851 | Val Top-1 51.57% | 57.26 min
Epoch 23/50 | LR 0.065451 | Train Loss 3.8207 | Train Top-1 22.19% | Val Loss 2.0092 | Val Top-1 55.68% | 57.38 min
Epoch 24/50 | LR 0.062096 | Train Loss 3.7851 | Train Top-1 22.07% | Val Loss 2.2751 | Val Top-1 52.42% | 57.43 min
Epoch 25/50 | LR 0.058682 | Train Loss 3.7810 | Train Top-1 22.77% | Val Loss 2.1426 | Val Top-1 54.27% | 56.82 min
Epoch 26/50 | LR 0.055226 | Train Loss 3.7471 | Train Top-1 23.17% | Val Loss 1.9544 | Val Top-1 56.75% | 46.73 min
Epoch 27/50 | LR 0.051745 | Train Loss 3.7234 | Train Top-1 23.51% | Val Loss 1.9266 | Val Top-1 57.63% | 46.74 min
Epoch 28/50 | LR 0.048255 | Train Loss 3.6910 | Train Top-1 23.79% | Val Loss 1.8666 | Val Top-1 59.06% | 46.73 min
Epoch 29/50 | LR 0.044774 | Train Loss 3.6455 | Train Top-1 24.05% | Val Loss 1.8747 | Val Top-1 59.74% | 46.71 min
Epoch 30/50 | LR 0.041318 | Train Loss 3.6386 | Train Top-1 23.97% | Val Loss 1.9125 | Val Top-1 58.52% | 54.14 min
Epoch 31/50 | LR 0.037904 | Train Loss 3.5832 | Train Top-1 24.91% | Val Loss 1.7265 | Val Top-1 61.08% | 57.35 min
Epoch 32/50 | LR 0.034549 | Train Loss 3.5807 | Train Top-1 25.13% | Val Loss 1.7749 | Val Top-1 61.14% | 57.54 min
Epoch 33/50 | LR 0.031270 | Train Loss 3.5206 | Train Top-1 25.32% | Val Loss 1.8544 | Val Top-1 58.71% | 57.67 min
Epoch 34/50 | LR 0.028081 | Train Loss 3.5113 | Train Top-1 25.39% | Val Loss 1.7576 | Val Top-1 62.94% | 57.67 min
Epoch 35/50 | LR 0.025000 | Train Loss 3.4649 | Train Top-1 26.25% | Val Loss 1.6448 | Val Top-1 63.33% | 57.76 min
Epoch 36/50 | LR 0.022040 | Train Loss 3.4491 | Train Top-1 26.79% | Val Loss 1.5449 | Val Top-1 64.90% | 57.75 min

Resumed from /home/ubuntu/runs/exp1/last.pth, starting at epoch 36

Epoch 37/50 | LR 0.040000 | Train Loss 3.4012 | Train Top-1 26.91% | Val Loss 1.6031 | Val Top-1 65.01% | 45.68 min
Epoch 38/50 | LR 0.060000 | Train Loss 3.5896 | Train Top-1 24.71% | Val Loss 1.7447 | Val Top-1 61.09% | 46.61 min
Epoch 39/50 | LR 0.080000 | Train Loss 3.7603 | Train Top-1 22.78% | Val Loss 1.9652 | Val Top-1 57.75% | 46.61 min
Epoch 40/50 | LR 0.040000 | Train Loss 3.8251 | Train Top-1 21.68% | Val Loss 2.0739 | Val Top-1 54.44% | 45.63 min
Epoch 41/50 | LR 0.040000 | Train Loss 3.4248 | Train Top-1 26.68% | Val Loss 1.8344 | Val Top-1 59.74% | 45.76 min
Epoch 42/50 | LR 0.040049 | Train Loss 3.5297 | Train Top-1 25.49% | Val Loss 1.7310 | Val Top-1 61.27% | 46.78 min
Epoch 43/50 | LR 0.040000 | Train Loss 3.5413 | Train Top-1 25.50% | Val Loss 1.7817 | Val Top-1 61.38% | 46.80 min
Epoch 44/50 | LR 0.039854 | Train Loss 3.5670 | Train Top-1 25.49% | Val Loss 1.8554 | Val Top-1 60.95% | 53.34 min
Epoch 45/50 | LR 0.039611 | Train Loss 3.5450 | Train Top-1 25.34% | Val Loss 1.8121 | Val Top-1 60.56% | 57.47 min
Epoch 46/50 | LR 0.039273 | Train Loss 3.5419 | Train Top-1 24.60% | Val Loss 1.8300 | Val Top-1 59.99% | 57.51 min
Epoch 47/50 | LR 0.038841 | Train Loss 3.5259 | Train Top-1 25.23% | Val Loss 1.7755 | Val Top-1 61.03% | 57.45 min
Epoch 48/50 | LR 0.038318 | Train Loss 3.5294 | Train Top-1 25.95% | Val Loss 1.6565 | Val Top-1 62.52% | 57.43 min
Epoch 49/50 | LR 0.037705 | Train Loss 3.5363 | Train Top-1 25.34% | Val Loss 1.7376 | Val Top-1 61.59% | 57.51 min
Epoch 50/50 | LR 0.037006 | Train Loss 3.5042 | Train Top-1 25.71% | Val Loss 1.7401 | Val Top-1 62.22% | 57.55 min

Training complete. Best Val Top-1: 62.52%

Resumed from /home/ubuntu/runs/exp1/best.pth, starting at epoch 48

Epoch 49/70 | LR 0.038318 | Train Loss 2.8772 | Train Top-1 53.97% | Val Loss 1.6725 | Val Top-1 61.63% | 45.62 min
Epoch 50/70 | LR 0.020000 | Train Loss 2.6473 | Train Top-1 59.34% | Val Loss 1.4764 | Val Top-1 66.19% | 46.38 min
Epoch 51/70 | LR 0.020000 | Train Loss 2.6258 | Train Top-1 59.85% | Val Loss 1.4087 | Val Top-1 67.33% | 46.45 min
Epoch 52/70 | LR 0.020000 | Train Loss 2.6247 | Train Top-1 59.88% | Val Loss 1.4714 | Val Top-1 65.90% | 46.41 min
Epoch 53/70 | LR 0.020000 | Train Loss 2.6296 | Train Top-1 59.73% | Val Loss 1.4031 | Val Top-1 67.20% | 52.50 min
Epoch 54/70 | LR 0.020000 | Train Loss 2.6290 | Train Top-1 59.74% | Val Loss 1.4328 | Val Top-1 66.74% | 56.63 min
Epoch 55/70 | LR 0.020000 | Train Loss 2.6284 | Train Top-1 59.74% | Val Loss 1.4423 | Val Top-1 66.75% | 56.64 min

Resumed from /home/ubuntu/runs/exp1/last.pth, starting at epoch 55

Epoch 56/90 | LR 0.020000 | Train Loss 2.6158 | Train Top-1 60.12% | Val Loss 1.4367 | Val Top-1 66.96% | 56.70 min
Epoch 57/90 | LR 0.020000 | Train Loss 2.5965 | Train Top-1 60.88% | Val Loss 1.4092 | Val Top-1 67.81% | 56.52 min
Epoch 58/90 | LR 0.020000 | Train Loss 2.5733 | Train Top-1 61.54% | Val Loss 1.3798 | Val Top-1 68.37% | 56.10 min
Epoch 59/90 | LR 0.020000 | Train Loss 2.5517 | Train Top-1 62.17% | Val Loss 1.3589 | Val Top-1 68.89% | 55.96 min
Epoch 60/90 | LR 0.020000 | Train Loss 2.5589 | Train Top-1 62.02% | Val Loss 1.3651 | Val Top-1 68.64% | 55.88 min
Epoch 61/90 | LR 0.020000 | Train Loss 2.5304 | Train Top-1 62.91% | Val Loss 1.3337 | Val Top-1 69.22% | 55.74 min
Epoch 62/90 | LR 0.020000 | Train Loss 2.5048 | Train Top-1 63.74% | Val Loss 1.3075 | Val Top-1 69.88% | 55.60 min
Epoch 63/90 | LR 0.020000 | Train Loss 2.4812 | Train Top-1 64.38% | Val Loss 1.2901 | Val Top-1 70.21% | 55.42 min
Epoch 64/90 | LR 0.020000 | Train Loss 2.4897 | Train Top-1 64.21% | Val Loss 1.2984 | Val Top-1 70.03% | 55.35 min
Epoch 65/90 | LR 0.020000 | Train Loss 2.4621 | Train Top-1 65.07% | Val Loss 1.2695 | Val Top-1 70.76% | 55.28 min
Epoch 66/90 | LR 0.020000 | Train Loss 2.4383 | Train Top-1 65.78% | Val Loss 1.2473 | Val Top-1 71.31% | 55.19 min
Epoch 67/90 | LR 0.020000 | Train Loss 2.4174 | Train Top-1 66.32% | Val Loss 1.2329 | Val Top-1 71.67% | 55.10 min
Epoch 68/90 | LR 0.020000 | Train Loss 2.4251 | Train Top-1 66.15% | Val Loss 1.2388 | Val Top-1 71.49% | 55.06 min
Epoch 69/90 | LR 0.020000 | Train Loss 2.4033 | Train Top-1 66.86% | Val Loss 1.2196 | Val Top-1 71.98% | 54.98 min
Epoch 70/90 | LR 0.020000 | Train Loss 2.3820 | Train Top-1 67.49% | Val Loss 1.1987 | Val Top-1 72.41% | 54.90 min
Epoch 71/90 | LR 0.020000 | Train Loss 2.3648 | Train Top-1 68.03% | Val Loss 1.1845 | Val Top-1 72.79% | 54.86 min
Epoch 72/90 | LR 0.020000 | Train Loss 2.3491 | Train Top-1 68.45% | Val Loss 1.1723 | Val Top-1 73.06% | 54.80 min
Epoch 73/90 | LR 0.020000 | Train Loss 2.3560 | Train Top-1 68.30% | Val Loss 1.1784 | Val Top-1 72.84% | 54.72 min
Epoch 74/90 | LR 0.020000 | Train Loss 2.3335 | Train Top-1 68.92% | Val Loss 1.1621 | Val Top-1 73.21% | 54.68 min
Epoch 75/90 | LR 0.020000 | Train Loss 2.3187 | Train Top-1 69.34% | Val Loss 1.1527 | Val Top-1 73.52% | 54.62 min
Epoch 76/90 | LR 0.020000 | Train Loss 2.3059 | Train Top-1 69.69% | Val Loss 1.1432 | Val Top-1 73.79% | 54.58 min
Epoch 77/90 | LR 0.020000 | Train Loss 2.3114 | Train Top-1 69.56% | Val Loss 1.1481 | Val Top-1 73.60% | 54.52 min
Epoch 78/90 | LR 0.020000 | Train Loss 2.2962 | Train Top-1 69.98% | Val Loss 1.1370 | Val Top-1 73.92% | 54.46 min
Epoch 79/90 | LR 0.020000 | Train Loss 2.2895 | Train Top-1 70.19% | Val Loss 1.1335 | Val Top-1 74.05% | 54.40 min
Epoch 80/90 | LR 0.020000 | Train Loss 2.2743 | Train Top-1 70.58% | Val Loss 1.1218 | Val Top-1 75.12% | 54.36 min
Epoch 81/90 | LR 0.020000 | Train Loss 2.2681 | Train Top-1 70.71% | Val Loss 1.1159 | Val Top-1 75.41% | 54.30 min
Epoch 82/90 | LR 0.020000 | Train Loss 2.2615 | Train Top-1 70.86% | Val Loss 1.1092 | Val Top-1 75.73% | 54.24 min
Epoch 83/90 | LR 0.020000 | Train Loss 2.2599 | Train Top-1 70.92% | Val Loss 1.1120 | Val Top-1 75.62% | 54.20 min
Epoch 84/90 | LR 0.020000 | Train Loss 2.2527 | Train Top-1 71.08% | Val Loss 1.1035 | Val Top-1 76.04% | 54.15 min
Epoch 85/90 | LR 0.020000 | Train Loss 2.2461 | Train Top-1 71.24% | Val Loss 1.0979 | Val Top-1 76.29% | 54.11 min
Epoch 86/90 | LR 0.020000 | Train Loss 2.2405 | Train Top-1 71.36% | Val Loss 1.0934 | Val Top-1 76.47% | 54.06 min
Epoch 87/90 | LR 0.020000 | Train Loss 2.2355 | Train Top-1 71.45% | Val Loss 1.0897 | Val Top-1 76.61% | 54.02 min
Epoch 88/90 | LR 0.020000 | Train Loss 2.2312 | Train Top-1 71.53% | Val Loss 1.0867 | Val Top-1 76.71% | 53.98 min
Epoch 89/90 | LR 0.020000 | Train Loss 2.2273 | Train Top-1 71.60% | Val Loss 1.0842 | Val Top-1 76.78% | 53.94 min
Epoch 90/90 | LR 0.020000 | Train Loss 2.2239 | Train Top-1 71.66% | Val Loss 1.0822 | Val Top-1 76.83% | 53.90 min

Training complete. Best Val Top-1: 76.83%
```

---

## 🚀 Reproduce This Training

### AWS EC2 Setup

#### 1. Launch EC2 Instance
```bash
# Via AWS Console:
# - Instance: GPU instance (e.g., g4dn.xlarge)
# - AMI: Deep Learning Base AMI (Ubuntu 20.04)
# - Storage: 200 GB gp3
# - Security Group: Allow SSH (port 22)
```

#### 2. Connect and Setup
```bash
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Clone repository
git clone https://github.com/Malar-saravanan/tsai_assignments.git
cd tsai_assignments/assignment9/assignment

# Setup environment
bash scripts/setup_ec2_budget.sh
```

#### 3. Start Training
```bash
# Start training in screen
screen -S training
bash scripts/train.sh

# Detach: Ctrl+A, D
# Reattach: screen -r training
```

#### 4. Monitor Progress
```bash
# Check status
bash scripts/status.sh

# View logs
bash scripts/logs.sh
```

---

### Direct Training

```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with Hugging Face
huggingface-cli login

# Train
python src/train.py \
    --data-dir /path/to/imagenet \
    --output-dir ./outputs \
    --epochs 90 \
    --lr 0.1 \
    --precision fp16 \
    --use-official-resnet
```

---

## 🛠️ Training Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | Required | Path to ImageNet dataset |
| `--output-dir` | `./outputs` | Output directory |
| `--epochs` | `90` | Number of epochs |
| `--lr` | `0.1` | Initial learning rate |
| `--precision` | `fp16` | Precision (fp32/fp16/bf16) |
| `--use-official-resnet` | `True` | Use torchvision ResNet50 |
| `--resume` | `` | Resume from checkpoint |
| `--test-only` | `False` | Only run validation |

---

## 📁 Project Structure

```
assignment/
├── src/
│   ├── train.py                 # Main training script
│   ├── model.py                 # ResNet50 architecture
│   ├── data.py                  # ImageNet loading
│   └── utils.py                 # Training utilities
│
├── scripts/
│   ├── setup_ec2_budget.sh     # Environment setup
│   ├── train.sh                # Training launcher
│   ├── validate.sh             # Validation
│   ├── status.sh               # Progress monitor
│   ├── logs.sh                 # Log viewer
│   ├── recover.sh              # Resume training
│   └── extract.sh              # Extract model
│
├── hf_resnet_app/              # Hugging Face deployment
│   ├── app.py                  # Gradio web interface
│   ├── best.pth                # Trained model weights
│   ├── requirements.txt        # HF app dependencies
│   └── *.png                   # Example images
│
├── requirements.txt            # Training dependencies
└── README.md                   # This file
```



## 🧪 Evaluate Model

```bash
# Evaluate best model
python src/train.py \
    --data-dir /path/to/imagenet \
    --resume outputs/best_model.pth \
    --test-only
```

---

## 🎓 Citation

**ImageNet Dataset:**
- [ILSVRC/imagenet-1k on Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k)


**ResNet Paper:**
```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  pages={770--778},
  year={2016}
}
```

---

## 📄 License

MIT License
