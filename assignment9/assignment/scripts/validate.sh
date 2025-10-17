#!/bin/bash

# Pre-deployment validation
echo "Validating training setup..."

# Check environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment ready"
else
    echo "✗ Virtual environment not found"
    echo "  Create with: python -m venv venv && source venv/bin/activate"
    exit 1
fi

# Check dependencies
python -c "import torch, torchvision, PIL; print('✓ Dependencies installed')" 2>/dev/null || {
    echo "✗ Missing dependencies"
    echo "  Install with: pip install -r requirements.txt"
    exit 1
}

# Test model creation
python -c "
import sys
sys.path.append('src')
from model import create_resnet50
model = create_resnet50(num_classes=1000)
total_params = sum(p.numel() for p in model.parameters())
print(f'✓ Model ready ({total_params:,} parameters)')
" || {
    echo "✗ Model creation failed"
    exit 1
}

# Test training components
python -c "
import torch
import sys
sys.path.append('src')
from model import create_resnet50
from utils import LabelSmoothingCrossEntropy
from torch.cuda.amp import autocast, GradScaler

model = create_resnet50(num_classes=1000)
criterion = LabelSmoothingCrossEntropy()
scaler = GradScaler()

# Test forward pass
x = torch.randn(2, 3, 224, 224)
y = torch.randint(0, 1000, (2,))

with autocast():
    output = model(x)
    loss = criterion(output, y)

print('✓ Training components ready')
" || {
    echo "✗ Training component test failed"
    exit 1
}

echo ""
echo "Validation complete - ready for EC2 deployment"
echo ""
echo "Next steps:"
echo "1. Launch g4dn.xlarge EC2 instance"
echo "2. Run: bash scripts/setup_ec2_budget.sh"
echo "3. Run: bash scripts/train.sh"
