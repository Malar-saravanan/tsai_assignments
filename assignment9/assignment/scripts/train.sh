#!/bin/bash

#  ResNet50 Training Script
# Target: 75%+ accuracy on ImageNet 1K within $15 budget

echo "Starting ResNet50 training..."
echo "Target: 75%+ top-1 accuracy"
echo "Budget: $15 (g4dn.xlarge)"
echo ""

# Configuration
DATA_DIR="/data/imagenet"
OUTPUT_DIR="outputs"
LOG_DIR="$OUTPUT_DIR/logs"

# Create directories
mkdir -p $OUTPUT_DIR $LOG_DIR

# Check prerequisites
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: ImageNet dataset not found at $DATA_DIR"
    echo "Run: bash scripts/setup_ec2_budget.sh"
    exit 1
fi

# Activate environment
source venv/bin/activate || {
    echo "Error: Virtual environment not found"
    echo "Run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
}

# Generate log filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRAIN_LOG="$LOG_DIR/training_${TIMESTAMP}.log"
VAL_LOG="$LOG_DIR/validation_${TIMESTAMP}.log"
COMBINED_LOG="$LOG_DIR/combined_${TIMESTAMP}.log"

echo "Starting training at $(date)"
echo "Training logs: $TRAIN_LOG"
echo "Validation logs: $VAL_LOG"
echo "Combined logs: $COMBINED_LOG"
echo "Checkpoints: $OUTPUT_DIR"
echo ""

# Start training with comprehensive logging
python src/train.py \
    --data-dir $DATA_DIR \
    --output-dir $OUTPUT_DIR \
    2>&1 | tee "$COMBINED_LOG" | \
    awk '
    # Split logs based on content
    /Train Loss|Train Acc|Epoch.*Results/ { print | "tee -a '"$TRAIN_LOG"'" }
    /Validation|Val.*Acc|Best Acc/ { print | "tee -a '"$VAL_LOG"'" }
    /Starting|Device|Target|EXCELLENT|SUCCESS/ { print | "tee -a '"$VAL_LOG"'" }
    { print }  # Print everything to stdout
    '

RESULT=$?

echo ""
echo "Training session completed at $(date)"
echo ""

if [ $RESULT -eq 0 ]; then
    echo "Status: SUCCESS"
    echo "Final model: $OUTPUT_DIR/best_model.pth"
    
    # Extract final accuracy from logs
    FINAL_ACC=$(grep "Best Acc@1" "$COMBINED_LOG" | tail -1 | grep -o '[0-9]\+\.[0-9]\+' | tail -1)
    if [ ! -z "$FINAL_ACC" ]; then
        echo "Final accuracy: ${FINAL_ACC}%"
        if (( $(echo "$FINAL_ACC >= 75.0" | bc -l) )); then
            echo "🎯 TARGET ACHIEVED: 75%+ accuracy reached!"
        fi
    fi
    
    # Create training summary
    cat > "$OUTPUT_DIR/training_summary.txt" << EOF
ResNet50 Training Summary
========================
Date: $(date)
Status: Completed Successfully
Final Accuracy: ${FINAL_ACC:-"Check logs"}%

Logs:
- Combined: $COMBINED_LOG
- Training: $TRAIN_LOG  
- Validation: $VAL_LOG

Checkpoints: $OUTPUT_DIR/
Best Model: $OUTPUT_DIR/best_model.pth
EOF

    echo "Training summary: $OUTPUT_DIR/training_summary.txt"
    
else
    echo "Status: FAILED (exit code $RESULT)"
    echo "Check logs for details:"
    echo "  Combined: $COMBINED_LOG"
    echo "  Training: $TRAIN_LOG"
    echo "  Validation: $VAL_LOG"
    echo ""
    echo "Recovery options:"
    echo "  bash scripts/recover.sh    # Fix issues and resume"
    echo "  bash scripts/status.sh     # Check current status"
    exit $RESULT
fi
