#!/bin/bash

# One-click training script for AFlow + ROLL Integration
# This script starts the ROLL-native multi-domain training

set -e  # Exit on error

echo "=================================================="
echo "AFlow + ROLL Multi-Domain Training"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if setup was run
if [ ! -f "setup_env.sh" ]; then
    print_error "Setup not completed. Please run ./setup.sh first"
    exit 1
fi

# Source environment
print_info "Loading environment..."
source setup_env.sh

# Check GPU availability
print_info "Checking GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    print_warn "No NVIDIA GPU detected. Training will be very slow."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    gpu_info=$(nvidia-smi --query-gpu=name,memory.free --format=csv,noheader | head -1)
    print_info "GPU: $gpu_info"
fi

# Stop any existing Ray instances
print_info "Stopping any existing Ray instances..."
ray stop &> /dev/null || true
sleep 2

# Create log directory
LOG_DIR="/home/username/logs/roll_native_training"
mkdir -p "$LOG_DIR"

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_$TIMESTAMP.log"

print_info "Starting ROLL-native training..."
print_info "Configuration: roll_native_config.yaml"
print_info "Log file: $LOG_FILE"
echo ""

print_info "Training details:"
echo "  - Model: Qwen-3-8B"
echo "  - Algorithm: GRPO (Group Relative Policy Optimization)"
echo "  - Domains: 6 (MATH, GSM8K, HumanEval, MBPP, HotpotQA, DROP)"
echo "  - Training samples: 902"
echo "  - Max steps: 1000"
echo "  - Batch size: 16"
echo ""

# Ask for confirmation
read -p "Start training? (Y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    print_info "Training cancelled"
    exit 0
fi

# Start training in background
print_info "Launching training process..."
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH

nohup python start_roll_native_training.py > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

print_info "Training started with PID: $TRAIN_PID"
echo ""

# Wait a moment for initialization
print_info "Initializing... (this may take 1-2 minutes)"
sleep 5

# Check if process is still running
if ps -p $TRAIN_PID > /dev/null; then
    print_info "Training process is running"
    echo ""
    print_info "Monitor training with:"
    echo "  tail -f $LOG_FILE"
    echo ""
    print_info "Or use:"
    echo "  watch -n 5 'tail -20 $LOG_FILE'"
    echo ""
    print_info "Check training progress:"
    echo "  grep -E '(step|loss|reward)' $LOG_FILE | tail -20"
    echo ""
    print_info "To stop training:"
    echo "  kill $TRAIN_PID"
    echo "  ray stop"
    echo ""

    # Show initial logs
    print_info "Initial training output:"
    echo "=================================================="
    sleep 10
    tail -50 "$LOG_FILE" | grep -v "metrics exporter\|rpc_code" || tail -20 "$LOG_FILE"
    echo "=================================================="
    echo ""

    print_info "Training is running in background (PID: $TRAIN_PID)"
    print_info "Logs: $LOG_FILE"
else
    print_error "Training process failed to start"
    print_error "Check logs for details: $LOG_FILE"
    tail -50 "$LOG_FILE"
    exit 1
fi
