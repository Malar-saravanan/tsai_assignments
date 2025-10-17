#!/bin/bash

# Training Status Monitor
echo "Training Status Report"
echo "====================="
echo "Time: $(date)"
echo ""

# System status
if command -v nvidia-smi &> /dev/null; then
    GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "%.1fGB/%.1fGB", $1/1024, $2/1024}')
    echo "GPU: ${GPU_USAGE}% utilization, $GPU_MEMORY memory"
else
    echo "GPU: Not available"
fi

DISK_USAGE=$(df -h . | tail -1 | awk '{print $4 " available"}')
echo "Disk: $DISK_USAGE"
echo ""

# Training progress
if [ -d "outputs" ]; then
    CHECKPOINT_COUNT=$(find outputs -name "checkpoint_epoch_*.pth" 2>/dev/null | wc -l)
    
    if [ $CHECKPOINT_COUNT -gt 0 ]; then
        LATEST_CHECKPOINT=$(ls -t outputs/checkpoint_epoch_*.pth 2>/dev/null | head -1)
        LATEST_EPOCH=$(basename "$LATEST_CHECKPOINT" | grep -o '[0-9]\+')
        PROGRESS=$((LATEST_EPOCH * 100 / 75))
        
        echo "Progress: Epoch $LATEST_EPOCH/75 (${PROGRESS}%)"
        
        # Check logs for accuracy
        LATEST_LOG=$(ls -t outputs/logs/training_*.log 2>/dev/null | head -1)
        if [ -f "$LATEST_LOG" ]; then
            BEST_ACC=$(grep "Best Acc@1" "$LATEST_LOG" | tail -1 | grep -o '[0-9]\+\.[0-9]\+' | tail -1)
            if [ ! -z "$BEST_ACC" ]; then
                echo "Best accuracy: ${BEST_ACC}%"
                
                if (( $(echo "$BEST_ACC >= 75.0" | bc -l) )); then
                    echo "Status: ✓ TARGET ACHIEVED!"
                elif (( $(echo "$BEST_ACC >= 70.0" | bc -l) )); then
                    echo "Status: On track for 75%+"
                else
                    echo "Status: Training in progress"
                fi
            fi
            
            # Check for recent errors
            ERROR_COUNT=$(tail -100 "$LATEST_LOG" | grep -i "error\|exception" | wc -l)
            if [ $ERROR_COUNT -gt 0 ]; then
                echo "Warnings: $ERROR_COUNT recent errors found"
            fi
        fi
        
        # Estimate time remaining
        if [ $LATEST_EPOCH -gt 5 ]; then
            EPOCHS_REMAINING=$((75 - LATEST_EPOCH))
            HOURS_REMAINING=$((EPOCHS_REMAINING / 4))  # ~4 epochs/hour
            if [ $HOURS_REMAINING -gt 0 ]; then
                echo "Estimated time remaining: ~${HOURS_REMAINING}h"
            fi
        fi
    else
        echo "Status: No checkpoints found - training not started"
    fi
else
    echo "Status: Training not started (no outputs directory)"
fi

echo ""
echo "Available commands:"
echo "  bash scripts/status.sh     - This status report"
echo "  bash scripts/recover.sh    - Fix issues and resume"
echo "  bash scripts/extract.sh    - Get best model from checkpoints"
