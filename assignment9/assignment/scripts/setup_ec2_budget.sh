#!/bin/bash

# EC2 Setup for ResNet50 Training
# Optimized for g4dn.xlarge (Tesla T4) under $15 budget

echo "Setting up EC2 for ResNet50 Training"
echo "Instance: g4dn.xlarge (Tesla T4)"
echo "Target: 75%+ accuracy in 18-20 hours"
echo "========================================="

# Update system
sudo apt-get update -y
sudo apt-get install -y wget unzip htop screen

# Create Python virtual environment (using system Python 3.8+ on Deep Learning AMI)
python3 -m venv venv
source venv/bin/activate
echo "Virtual environment created and activated"

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Setup ImageNet data directory
mkdir -p /data/imagenet
echo "📁 Created /data/imagenet directory"

# Download ImageNet from AWS Open Data (no registration required)
echo "Downloading ImageNet 1K dataset..."
echo "This will take 30-60 minutes depending on connection speed"

# Install AWS CLI if not present
if ! command -v aws &> /dev/null; then
    pip install awscli
fi

# Download training set (most time-consuming)
echo "Downloading training set..."
aws s3 sync s3://imagenet-original/train/ /data/imagenet/train/ --no-sign-request --quiet

# Download validation set
echo "Downloading validation set..."  
aws s3 sync s3://imagenet-original/val/ /data/imagenet/val/ --no-sign-request --quiet

echo "ImageNet dataset downloaded successfully"
echo "Dataset size: ~150GB (1.2M training + 50K validation images)"

# Verify setup
echo ""
echo "Verifying setup..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

if [ -d "/data/imagenet/train" ] && [ -d "/data/imagenet/val" ]; then
    TRAIN_CLASSES=$(ls /data/imagenet/train | wc -l)
    VAL_IMAGES=$(find /data/imagenet/val -name "*.JPEG" | wc -l)
    echo "Training classes: $TRAIN_CLASSES (expected: 1000)"
    echo "Validation images: $VAL_IMAGES (expected: ~50,000)"
fi

echo ""
echo "Setup complete! Ready for training."
echo "Estimated cost: $9-11 total (18-20 hours @ $0.526/hour)"
echo "Target: 75%+ top-1 accuracy"
echo ""
echo "Next steps:"
echo "1. Validate setup: bash scripts/validate.sh"
echo "2. Start training: bash scripts/train.sh"
