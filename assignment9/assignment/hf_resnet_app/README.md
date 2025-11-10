---
title: ResNet50 ImageNet-1K Classifier
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
---

# ResNet50 ImageNet-1K Classifier

ResNet50 trained from scratch on ImageNet 1K achieving **76.83% top-1 accuracy**.

## About

This model was trained from scratch (no pre-trained weights) on the full ImageNet-1K dataset, achieving competitive performance using advanced training techniques including:
- AutoAugment (ImageNet policy)
- RandomErasing
- Mixup augmentation
- Label smoothing
- Mixed precision (FP16) training
- Cosine learning rate schedule with warmup

## Links

- **GitHub Repository:** [tsai_assignments/assignment9](https://github.com/Malar-saravanan/tsai_assignments/tree/session9/assignment9/assignment)
- **Dataset:** [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k)
- **Reference:** [godsofheaven/Resnet50-from-Scratch-on-Imagenet-1K](https://github.com/godsofheaven/Resnet50-from-Scratch-on-Imagenet-1K)

## Performance

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 76.83% |
| Top-5 Accuracy | 93.2% |
| Model Parameters | 25.6M |
| Model Size (CPU) | 98 MB |
