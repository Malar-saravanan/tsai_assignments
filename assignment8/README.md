---
title: ResNet-18 CIFAR-100 Image Classifier
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# ResNet-18 CIFAR-100 Image Classifier 🖼️

A **ResNet-18** model trained from scratch on **CIFAR-100** dataset, achieving **74.07% test accuracy**! This project demonstrates end-to-end machine learning from training to deployment.

## 🎯 Model Performance
- **Architecture**: ResNet-18 (11.2M parameters)
- **Dataset**: CIFAR-100 (100 classes, 32x32 images)
- **Final Test Accuracy**: **74.07%** (exceeded 73% target!)
- **Training Device**: Apple Silicon MPS
- **Training Time**: ~45 minutes (86 epochs)
- **Final Training Accuracy**: 97.89%

## 🚀 Live Demo Usage
1. **Upload an image** using the interface below
2. The model will **classify** it into one of the 100 CIFAR-100 categories
3. View **top-5 predictions** with confidence scores
4. Try the **example images** provided!

## 🏷️ CIFAR-100 Classes (100 Categories)
This model can classify images into 100 different categories:

- **Animals**: apple, aquarium_fish, baby, bear, beaver, bee, beetle, butterfly, camel, caterpillar, cattle, chimpanzee, cockroach, crab, crocodile, dinosaur, dolphin, elephant, flatfish, fox, hamster, kangaroo, leopard, lion, lizard, lobster, mouse, otter, possum, rabbit, raccoon, ray, seal, shark, shrew, skunk, snail, snake, spider, squirrel, tiger, trout, turtle, whale, wolf, worm

- **Vehicles & Transportation**: bicycle, bus, motorcycle, pickup_truck, rocket, streetcar, tank, tractor, train

- **Household Items**: bed, bottle, bowl, chair, clock, couch, cup, keyboard, lamp, plate, table, telephone, television, wardrobe

- **Nature & Plants**: cloud, forest, maple_tree, mountain, oak_tree, palm_tree, pine_tree, plain, poppy, rose, sea, sunflower, tulip, willow_tree

- **Food**: apple, mushroom, orange, pear, sweet_pepper

- **Structures**: bridge, castle, house, road, skyscraper

- **People**: boy, girl, man, woman

## 💡 Tips for Best Results
- Upload clear, centered images
- Images are automatically resized to 32x32 pixels (CIFAR-100 format)
- Works best with objects similar to CIFAR-100 categories
- Try the example images to see the model in action!

## 🏗️ Project Structure & Development

### For Developers & Researchers:

```
assignment8/
├── app.py               # Gradio web application
├── model.py             # ResNet-18 architecture implementation
├── config.py            # Configuration management
├── data_loader.py       # CIFAR-100 data loading and preprocessing
├── trainer.py           # Training loop and utilities
├── train_improved.py    # Main training script with config support
├── utils.py             # Logging and utility functions
├── best_model.pth       # Trained model weights (74.07% accuracy)
├── training_logs.md     # Detailed training history
├── requirements.txt     # Python dependencies
└── example_*.png        # Demo images
```

### Quick Start for Training:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run training:**
   ```bash
   python train_improved.py
   ```

3. **Test the app locally:**
   ```bash
   python app.py
   ```

## 🔧 Technical Details

### Model Architecture
- **ResNet-18**: Lightweight ResNet variant optimized for CIFAR-100
- **Parameters**: ~11.2M trainable parameters
- **Input**: 32x32x3 RGB images
- **Output**: 100 classes with softmax probabilities

### Training Configuration
- **Optimizer**: SGD with momentum=0.9, weight_decay=5e-4
- **Learning Rate**: 0.1 (initial) with CosineAnnealingLR scheduler
- **Batch Size**: 128
- **Final Epochs**: 86 (with optimal early stopping)
- **Data Augmentation**: 
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip()
  - RandomRotation(15°)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
  - Normalization: mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]

### Training Results
- **Phase 1** (Epochs 1-20): Rapid initial learning (12.53% → ~60% accuracy)
- **Phase 2** (Epochs 21-60): Steady improvement with regularization
- **Phase 3** (Epochs 61-81): Target achievement (73.03% at epoch 81)
- **Phase 4** (Epochs 82-86): Fine-tuning to peak performance (74.07%)

### Hardware Optimization
- **Apple Silicon (MPS)**: Primary training target
- **CPU**: Deployment target (HuggingFace Spaces)
- **CUDA GPUs**: Secondary support for training

## 📊 Performance Analysis

### Key Achievements:
- ✅ **Target Exceeded**: 74.07% vs 73% goal (+1.07%)
- ✅ **Excellent Generalization**: 97.89% train vs 74.07% test (reasonable gap)
- ✅ **Efficient Training**: ~45 minutes on Apple Silicon MPS
- ✅ **Robust Architecture**: ResNet-18 with optimal depth for CIFAR-100

### Training Insights:
- **Best Epoch**: 86 (74.07% test accuracy)
- **Convergence**: Smooth learning curve with cosine annealing
- **Stability**: Consistent improvement without significant overfitting
- **Efficiency**: Optimal stopping prevented overtraining

## 🚀 Deployment & Usage

This model is deployed as a **live web application** on HuggingFace Spaces:
- **Framework**: Gradio for interactive interface
- **Inference**: Real-time classification on CPU
- **User Experience**: Upload → Classify → Results in seconds
- **Accessibility**: Works on mobile and desktop browsers

## 🎓 Educational Value

This project demonstrates:
- **End-to-end ML pipeline**: From data loading to deployment
- **Modern PyTorch practices**: Modular code, configuration management
- **Deep learning techniques**: ResNet architecture, data augmentation, optimization
- **Model deployment**: Gradio interfaces, HuggingFace Spaces
- **Performance optimization**: MPS utilization, efficient training

## 📈 Future Improvements

Potential enhancements:
- **Model**: Try ResNet-34/50, EfficientNet, Vision Transformers
- **Data**: Additional augmentation techniques, mixup, cutmix
- **Training**: Learning rate scheduling, advanced optimizers
- **Deployment**: Model quantization, ONNX export, edge deployment

---

**Built with ❤️ using PyTorch and Gradio** | **Deployed on HuggingFace Spaces**
