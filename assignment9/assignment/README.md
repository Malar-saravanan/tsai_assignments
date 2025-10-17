# ResNet50 ImageNet Training - 75%+ Accuracy

Production-ready ResNet50 implementation optimized for 75%+ top-1 accuracy on ImageNet 1K within $15 EC2 budget.

## 🎯 Performance Targets
- **Accuracy**: 75%+ top-1 on ImageNet 1K (1000 classes)  
- **Time**: 18-20 hours on g4dn.xlarge (Tesla T4)
- **Cost**: Under $15 (target) / $18 (maximum)
- **Auto-stop**: Training stops at 77%+ to save budget

## 🏗️ ResNet50 Architecture

Our implementation features a **custom ResNet50 with 25.6M parameters** optimized for ImageNet classification:

### **Core Architecture**
```
Input: 224×224×3 RGB images
├── Conv1: 7×7 conv, 64 filters, stride=2 → 112×112×64
├── BatchNorm + ReLU → 112×112×64
├── MaxPool: 3×3, stride=2 → 56×56×64
├── Layer1: 3×Bottleneck(64→256) → 56×56×256  
├── Layer2: 4×Bottleneck(128→512) → 28×28×512
├── Layer3: 6×Bottleneck(256→1024) → 14×14×1024
├── Layer4: 3×Bottleneck(512→2048) → 7×7×2048
├── AdaptiveAvgPool2d → 1×1×2048
└── Linear Layer → 1000 classes (ImageNet)
```

### **Bottleneck Block Design**
Each bottleneck block implements:
- **1×1 Conv** (channel reduction) → **3×3 Conv** (spatial features) → **1×1 Conv** (channel expansion)
- **Residual connections** for gradient flow
- **Batch normalization** after each convolution
- **ReLU activation** functions

### **Performance Optimizations**
- **Channels-last memory format**: 10-15% speed boost on Tesla T4
- **Mixed precision (FP16)**: 2x memory efficiency, faster training
- **PyTorch compilation**: Additional 10-20% performance gain
- **Optimal batch size (128)**: Maximizes Tesla T4 16GB VRAM usage

## 🚀 EC2 Setup Guide

### **Step 1: Launch EC2 Instance**
```bash
# Recommended instance: g4dn.xlarge
# - Tesla T4 GPU (16GB VRAM)
# - 4 vCPUs, 16GB RAM
# - Cost: ~$0.526/hour
# - Storage: 100GB EBS GP3

# AMI: Deep Learning AMI Ubuntu 20.04
# Security Group: Allow SSH (port 22)
```

### **Step 2: Connect and Setup Environment**
```bash
# Connect to your instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Clone your repository
git clone <your-repo-url>
cd assignment

# Run automated setup (downloads ImageNet, installs dependencies)
bash scripts/setup_ec2_budget.sh
```

### **Step 3: Start Training**
```bash
# Validate setup first
bash scripts/validate.sh

# Start training with comprehensive logging
bash scripts/train.sh

# Monitor progress (in another terminal)
bash scripts/status.sh
```

### **Step 4: Monitor Training**
```bash
# Check real-time status
bash scripts/status.sh

# View training logs
bash scripts/logs.sh

# Recovery if needed
bash scripts/recover.sh
```

## 🧠 Training Strategy

### **Optimization Techniques**
1. **Label Smoothing (0.1)**: Prevents overconfidence and improves generalization
2. **Mixup Augmentation (0.2)**: Mixes training examples for better robustness
3. **Cosine Learning Rate Schedule**: Smooth decay from 0.1 to 0.00005
4. **Mixed Precision Training**: FP16 for 2x memory efficiency and speed
5. **Advanced Data Augmentation**: RandomResizedCrop, ColorJitter, normalization

### **Hardware Optimization**
- **Channels-last memory format**: Optimized for modern GPUs
- **Optimal batch size (128)**: Maximizes Tesla T4 utilization
- **Gradient scaling**: Prevents underflow in mixed precision
- **Multi-worker data loading**: Overlaps I/O with computation

### **Budget Management**
- **Automatic checkpointing**: Every epoch after epoch 10
- **Early stopping**: Stops at 77%+ accuracy to save budget
- **Progress monitoring**: Warnings if accuracy falls behind schedule
- **Cost tracking**: Real-time budget estimation

## 📊 Logging and Monitoring

### **Log Files**
- **Training logs**: `outputs/logs/training_TIMESTAMP.log` - Training metrics
- **Validation logs**: `outputs/logs/validation_TIMESTAMP.log` - Validation results  
- **Combined logs**: `outputs/logs/combined_TIMESTAMP.log` - Complete session

### **Structured Output Format**
```
VALIDATION_RESULTS Epoch 25/75:
  Val_Acc@1: 72.450%
  Best_Acc@1: 73.120% 
  Val_Time: 2.3min
  Total_Time: 8.5h
  Est_Cost: $4.47
```

### **Checkpoints**
- **Regular checkpoints**: `outputs/checkpoint_epoch_N.pth`
- **Best model**: `outputs/best_model.pth` (highest accuracy)
- **Training state**: Optimizer, scaler, epoch info included

## 🛠️ Essential Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `validate.sh` | Pre-deployment validation | `bash scripts/validate.sh` |
| `setup_ec2_budget.sh` | EC2 environment setup | `bash scripts/setup_ec2_budget.sh` |
| `train.sh` | Main training with logging | `bash scripts/train.sh` |
| `status.sh` | Real-time progress monitor | `bash scripts/status.sh` |
| `logs.sh` | Log analysis and metrics | `bash scripts/logs.sh` |
| `recover.sh` | Issue recovery and resume | `bash scripts/recover.sh` |
| `extract.sh` | Extract best model | `bash scripts/extract.sh` |

## 📁 Project Structure

```
assignment/
├── src/
│   ├── model.py          # ResNet50 architecture
│   ├── data.py           # ImageNet data loading
│   ├── utils.py          # Training utilities
│   └── train.py          # Main training script
├── scripts/
│   ├── validate.sh       # Pre-deployment validation
│   ├── setup_ec2_budget.sh # EC2 environment setup
│   ├── train.sh          # Training with logging
│   ├── status.sh         # Progress monitoring
│   ├── logs.sh           # Log analysis
│   ├── recover.sh        # Recovery utilities
│   └── extract.sh        # Model extraction
├── requirements.txt      # Python dependencies
├── app.py               # Gradio demo (optional)
└── README.md            # This file
```

## 🎯 Expected Results

### **Accuracy Progression**
- **Epoch 10**: ~35-40% (initial learning)
- **Epoch 25**: ~55-60% (rapid improvement)
- **Epoch 50**: ~70-72% (approaching target)
- **Epoch 65**: ~74-75% (target range)
- **Epoch 75**: ~75-77% (final accuracy)

### **Cost Breakdown**
- **Instance**: g4dn.xlarge @ $0.526/hour
- **Training time**: 18-20 hours
- **Total cost**: $9.47-$10.52 (well under $15 budget)
- **Early stop savings**: If 77%+ reached early

### **Professional Features**
- ✅ Comprehensive error handling and recovery
- ✅ Automated checkpoint management
- ✅ Real-time cost and progress tracking
- ✅ Professional logging and monitoring
- ✅ Fallback strategies for edge cases
- ✅ Clean, production-ready codebase
bash scripts/recover.sh
```
- Optimized batch size (128) for Tesla T4 GPU

**Training Strategy:**
- Label smoothing (0.1) + Mixup (0.2) for +1-2% accuracy boost
- Cosine learning rate schedule with warmup (proven effective)
- Advanced data augmentation with bicubic interpolation
- Strategic checkpointing every epoch after epoch 10

## � Comprehensive Logging & Monitoring

**Training Logs (Auto-generated):**
- `outputs/logs/combined_TIMESTAMP.log` - Complete training session
- `outputs/logs/training_TIMESTAMP.log` - Training-specific metrics
- `outputs/logs/validation_TIMESTAMP.log` - Validation results and accuracy
- `outputs/training_summary.txt` - Final training summary

**Monitoring Commands:**
- `bash scripts/status.sh` - Current training status and progress
- `bash scripts/logs.sh` - Detailed log analysis and metrics extraction
- `bash scripts/recover.sh` - Fix issues and resume training
- `bash scripts/extract.sh` - Extract best model from checkpoints

**Log Structure:**
```
VALIDATION_RESULTS Epoch X/75:
  Val_Acc@1: XX.XXX%
  Best_Acc@1: XX.XXX%
  Val_Time: X.Xmin
  Total_Time: X.XXh
  Est_Cost: $X.XX
```

## 🛠️ Essential Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/validate.sh` | Pre-deployment validation | Test setup before EC2 |
| `scripts/setup_ec2_budget.sh` | EC2 environment setup | Download ImageNet, install deps |
| `scripts/train.sh` | **Main training script** | Start ResNet50 training |
| `scripts/status.sh` | Training progress monitor | Check current status |
| `scripts/logs.sh` | Log analysis tool | Extract metrics from logs |
| `scripts/recover.sh` | Issue recovery | Fix problems and resume |
| `scripts/extract.sh` | Model extraction | Get final trained model |

**Robust Training Features:**
- **Automatic checkpointing** every epoch (after epoch 10)
- **Best model tracking** with comprehensive metadata
- **Budget monitoring** with cost estimation ($0.526/hour on g4dn.xlarge)
- **Error recovery** with batch size reduction fallbacks
- **Progress warnings** at epochs 30, 50 with intervention suggestions
- **Early stopping** at 77%+ accuracy to save budget

## Assignment Requirements
✅ ResNet50 from scratch (no pre-trained weights)  
✅ ImageNet 1K training (1000 classes, automatic download)  
✅ EC2 deployment with cost optimization  
✅ Production-ready implementation with robust fallbacks
