# Efficient CNN Assignment: MNIST Classification

**Assignment Goal**: Achieve 99.4% validation accuracy consistently in the last few epochs with ≤8000 parameters in ≤15 epochs using iterative learning approach.

This project demonstrates systematic CNN development following iterative learning principles. Each model builds upon the previous one, addressing specific limitations and progressively improving towards the target.

## Model Development Strategy

 we use an iterative approach:
1. **Model_1**: Establish baseline architecture
2. **Model_2**: Add regularization techniques  
3. **Model_3**: Optimize training dynamics

---

## Model_1: Basic Skeleton Setup

### Target
Get the basic CNN skeleton right with minimal parameters. Establish baseline architecture with conv layers, batch norm, and basic structure. No regularization or fancy techniques - just core CNN building blocks.

### Results
- **Parameters**: 7,450
- **Best Training Accuracy**: 99.08%
- **Best Test Accuracy**: 99.04%


### Analysis
The basic model shows good learning capacity with minimal overfitting (train: 99.08% vs test: 99.04%). The small gap indicates decent generalization, but the test accuracy of 99.04% is still short of our 99.4% target. The architecture has sufficient capacity, so we need regularization to improve generalization further.

### Architecture
```
Input (1x28x28) 
→ Conv2d(1→8, 3x3, pad=1) + BN + ReLU     # RF=3, 28x28
→ Conv2d(8→12, 3x3, pad=1) + BN + ReLU    # RF=5, 28x28  
→ MaxPool2d(2x2)                          # RF=6, 14x14
→ Conv2d(12→14, 3x3, pad=1) + BN + ReLU   # RF=10, 14x14
→ MaxPool2d(2x2)                          # RF=12, 7x7
→ Conv2d(14→16, 3x3) + BN + ReLU          # RF=20, 5x5
→ Conv2d(16→18, 3x3) + BN + ReLU          # RF=28, 3x3
→ Conv2d(18→10, 1x1)                      # RF=28, 3x3
→ GAP → FC(10)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.01, weight_decay=1e-4)
- **Scheduler**: None
- **Augmentation**: None
- **Loss**: NLLLoss
- **Epochs**: 15


---

## Model_2: Add Regularization

### Target
Reduce overfitting by adding dropout after pooling layers. Add data augmentation (rotation) to make training harder and improve generalization. Keep same architecture but add regularization techniques.


### Results
- **Parameters**: 7,450 (same as Model_1)
- **Best Training Accuracy**: 98.51%
- **Best Test Accuracy**: 99.21%


### Analysis
Regularization is working! The model now generalizes better with test accuracy (99.21%) higher than training (98.51%), showing the dropout and augmentation are preventing overfitting. The +0.17% improvement over Model_1 demonstrates that regularization helps. The model is slightly under-fitting, which provides room for training optimization.


### Architecture
```
Same as Model_1 but with:
→ MaxPool2d(2x2) → Dropout2d(0.1)         # After first pooling
→ MaxPool2d(2x2) → Dropout2d(0.1)         # After second pooling
```

### Training Configuration
- **Optimizer**: Adam (lr=0.01, weight_decay=1e-4)
- **Scheduler**: None
- **Augmentation**: RandomRotation(±6°)
- **Loss**: NLLLoss
- **Epochs**: 15

---

## Model_3: Optimize Training Dynamics

### Target
Add LR scheduler for better convergence and stability. Fine-tune dropout to balance regularization and capacity. Achieve consistent 99.4%+ accuracy in last few epochs.

### Results
- **Parameters**: 7,450 (same architecture)
- **Best Training Accuracy**: 99.18%
- **Best Test Accuracy**: 99.38%

### Analysis
The LR scheduler (StepLR) and reduced dropout (0.05) improved stability and pushed accuracy higher. We're getting very close to 99.4% (best: 99.38%) but missing the consistency requirement. The model shows stable training with good convergence around 99.3%. Need more capacity or advanced training techniques to cross 99.4% barrier consistently.

### Architecture
```
Same as Model_2 but with:
→ MaxPool2d(2x2) → Dropout2d(0.05)        # Reduced dropout
→ MaxPool2d(2x2) → Dropout2d(0.05)        # Reduced dropout
```

### Training Configuration
- **Optimizer**: Adam (lr=0.01, weight_decay=1e-4)
- **Scheduler**: StepLR (step_size=5, gamma=0.5)
- **Augmentation**: RandomRotation(±6°)
- **Loss**: NLLLoss
- **Epochs**: 15

---

## Receptive Field Calculations

All models follow the same RF progression:
- **Layer 1**: RF = 3 (3x3 conv)
- **Layer 2**: RF = 5 (3x3 conv)  
- **Pool 1**: RF = 6 (2x2 maxpool)
- **Layer 3**: RF = 10 (3x3 conv)
- **Pool 2**: RF = 12 (2x2 maxpool)
- **Layer 4**: RF = 20 (3x3 conv, no padding)
- **Layer 5**: RF = 28 (3x3 conv, no padding)
- **Final**: RF = 28 (1x1 conv)

The final RF of 28 is optimal for MNIST (28x28 images) as it covers the entire input space perfectly.

---

## Progressive Results Summary

| Model | Parameters | Best Accuracy | Key Innovation |
|-------|------------|---------------|----------------|
| Model_1 | 7,450 | 99.04% | Basic CNN skeleton |
| Model_2 | 7,450 | 99.21% | Dropout + Augmentation |
| Model_3 | 7,450 | 99.38% | LR Scheduler |

---

## Key Learnings

1. **Architecture First**: Establish solid skeleton before optimization
2. **Regularization Works**: Dropout + augmentation crucial for generalization
3. **Training Dynamics Matter**: LR scheduling significantly improves convergence
4. **Iterative Approach**: Each model addresses specific limitations of the previous one
5. **Parameter Efficiency**: Achieved high accuracy with <8K parameters using proper CNN techniques

---

## File Structure

```
├── model.py          # All 3 model definitions with Target/Result/Analysis
├── train.py          # Training logic and data loaders
├── test.ipynb        # Experiment logs and results
└── README.md         # This comprehensive documentation
```

## How to Run

```bash
# Install dependencies
pip install torch torchvision

# Train all models
python train.py

# Or run individual experiments in test.ipynb
jupyter notebook test.ipynb
```

---
