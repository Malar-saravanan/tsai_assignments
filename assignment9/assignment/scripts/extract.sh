#!/bin/bash

# Extract Best Model from Checkpoints
echo "Extracting best model..."

OUTPUT_DIR="outputs"
EXTRACT_DIR="final_model"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: No outputs directory found"
    exit 1
fi

# Find checkpoints
CHECKPOINTS=$(find $OUTPUT_DIR -name "checkpoint_epoch_*.pth" 2>/dev/null | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "Error: No checkpoints found"
    exit 1
fi

mkdir -p $EXTRACT_DIR

# Check if best_model.pth exists
if [ -f "$OUTPUT_DIR/best_model.pth" ]; then
    echo "Using saved best model..."
    cp "$OUTPUT_DIR/best_model.pth" "$EXTRACT_DIR/model.pth"
    
    # Extract accuracy from logs
    LATEST_LOG=$(ls -t $OUTPUT_DIR/logs/training_*.log 2>/dev/null | head -1)
    if [ -f "$LATEST_LOG" ]; then
        ACCURACY=$(grep "Best Acc@1" "$LATEST_LOG" | tail -1 | grep -o '[0-9]\+\.[0-9]\+' | tail -1)
        if [ ! -z "$ACCURACY" ]; then
            echo "Model accuracy: ${ACCURACY}%"
        fi
    fi
else
    echo "No best_model.pth found, using latest checkpoint..."
    LATEST_CHECKPOINT=$(ls -t $OUTPUT_DIR/checkpoint_epoch_*.pth | head -1)
    cp "$LATEST_CHECKPOINT" "$EXTRACT_DIR/model.pth"
    EPOCH=$(basename "$LATEST_CHECKPOINT" | grep -o '[0-9]\+')
    echo "Using checkpoint from epoch $EPOCH"
fi

# Create model info
cat > "$EXTRACT_DIR/model_info.txt" << EOF
ResNet50 ImageNet Model
=====================
Date: $(date)
Architecture: ResNet50 (25.6M parameters)
Dataset: ImageNet 1K (1000 classes)
Target: 75%+ top-1 accuracy

Training Features:
- Label smoothing (0.1)
- Mixup augmentation (0.2)
- Mixed precision training
- Cosine learning rate schedule
- Advanced data augmentation

Model file: model.pth
EOF

echo ""
echo "Model extracted to: $EXTRACT_DIR/model.pth"
echo "Model info: $EXTRACT_DIR/model_info.txt"

# Verify model can be loaded
python -c "
import sys
sys.path.append('src')
import torch
from model import create_resnet50

try:
    model = create_resnet50(num_classes=1000)
    checkpoint = torch.load('$EXTRACT_DIR/model.pth', map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print('✓ Model loaded successfully')
except Exception as e:
    print(f'✗ Model loading failed: {e}')
    exit(1)
"
