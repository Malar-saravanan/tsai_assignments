#!/bin/bash

# Log Analysis Tool
echo "Training Log Analysis"
echo "===================="

LOG_DIR="outputs/logs"

if [ ! -d "$LOG_DIR" ]; then
    echo "No logs directory found"
    exit 1
fi

# Find latest logs
LATEST_COMBINED=$(ls -t $LOG_DIR/combined_*.log 2>/dev/null | head -1)
LATEST_TRAIN=$(ls -t $LOG_DIR/training_*.log 2>/dev/null | head -1)  
LATEST_VAL=$(ls -t $LOG_DIR/validation_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_COMBINED" ]; then
    echo "No training logs found"
    exit 1
fi

echo "Latest logs:"
echo "  Combined: $(basename $LATEST_COMBINED)"
[ ! -z "$LATEST_TRAIN" ] && echo "  Training: $(basename $LATEST_TRAIN)"
[ ! -z "$LATEST_VAL" ] && echo "  Validation: $(basename $LATEST_VAL)"
echo ""

# Training Progress
echo "Training Progress:"
echo "=================="
if [ -f "$LATEST_COMBINED" ]; then
    # Current epoch
    CURRENT_EPOCH=$(grep "Epoch [0-9]*/75" "$LATEST_COMBINED" | tail -1 | grep -o "Epoch [0-9]*" | grep -o "[0-9]*")
    if [ ! -z "$CURRENT_EPOCH" ]; then
        PROGRESS=$((CURRENT_EPOCH * 100 / 75))
        echo "Current epoch: $CURRENT_EPOCH/75 (${PROGRESS}%)"
    fi
    
    # Latest training metrics
    LATEST_TRAIN_ACC=$(grep "Train Acc" "$LATEST_COMBINED" | tail -1 | grep -o "[0-9]*\.[0-9]*%" | head -1)
    LATEST_TRAIN_LOSS=$(grep "Train Loss" "$LATEST_COMBINED" | tail -1 | grep -o "Loss [0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*")
    
    [ ! -z "$LATEST_TRAIN_ACC" ] && echo "Latest train acc: $LATEST_TRAIN_ACC"
    [ ! -z "$LATEST_TRAIN_LOSS" ] && echo "Latest train loss: $LATEST_TRAIN_LOSS"
fi
echo ""

# Validation Results
echo "Validation Results:"
echo "=================="
if [ -f "$LATEST_COMBINED" ]; then
    echo "Recent validation accuracies:"
    grep "Val_Acc@1:" "$LATEST_COMBINED" | tail -5 | while read line; do
        EPOCH=$(echo "$line" | grep -o "Epoch [0-9]*" | grep -o "[0-9]*")
        ACC=$(echo "$line" | grep -o "[0-9]*\.[0-9]*%")
        echo "  Epoch $EPOCH: $ACC"
    done
    
    # Best accuracy
    BEST_ACC=$(grep "Best_Acc@1:" "$LATEST_COMBINED" | tail -1 | grep -o "[0-9]*\.[0-9]*")
    if [ ! -z "$BEST_ACC" ]; then
        echo ""
        echo "Best accuracy so far: ${BEST_ACC}%"
        
        if (( $(echo "$BEST_ACC >= 75.0" | bc -l) )); then
            echo "Status: ✓ TARGET ACHIEVED!"
        elif (( $(echo "$BEST_ACC >= 70.0" | bc -l) )); then
            echo "Status: On track for target"
        else
            echo "Status: Training in progress"
        fi
    fi
fi
echo ""

# Cost tracking
echo "Cost Analysis:"
echo "=============="
if [ -f "$LATEST_COMBINED" ]; then
    LATEST_COST=$(grep "Est_Cost:" "$LATEST_COMBINED" | tail -1 | grep -o "\$[0-9]*\.[0-9]*")
    LATEST_TIME=$(grep "Total_Time:" "$LATEST_COMBINED" | tail -1 | grep -o "[0-9]*\.[0-9]*h")
    
    [ ! -z "$LATEST_COST" ] && echo "Current cost: $LATEST_COST"
    [ ! -z "$LATEST_TIME" ] && echo "Training time: $LATEST_TIME" 
    
    if [ ! -z "$LATEST_COST" ]; then
        COST_NUM=$(echo "$LATEST_COST" | grep -o "[0-9]*\.[0-9]*")
        if (( $(echo "$COST_NUM <= 15.0" | bc -l) )); then
            echo "Budget status: ✓ Within $15 target"
        elif (( $(echo "$COST_NUM <= 18.0" | bc -l) )); then
            echo "Budget status: ⚠ Within $18 limit"
        else
            echo "Budget status: ❌ Over budget"
        fi
    fi
fi
echo ""

# Error Analysis
echo "Error Analysis:"
echo "==============="
if [ -f "$LATEST_COMBINED" ]; then
    ERROR_COUNT=$(grep -i "error\|exception\|failed" "$LATEST_COMBINED" | wc -l)
    if [ $ERROR_COUNT -gt 0 ]; then
        echo "Found $ERROR_COUNT error(s) in logs"
        echo "Recent errors:"
        grep -i "error\|exception\|failed" "$LATEST_COMBINED" | tail -3 | sed 's/^/  /'
    else
        echo "No errors found in recent logs"
    fi
fi

echo ""
echo "Full logs available in: $LOG_DIR"
