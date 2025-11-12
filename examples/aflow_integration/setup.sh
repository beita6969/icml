#!/bin/bash

# AFlow + ROLL Integration Setup Script
# This script automatically sets up the environment and dependencies

set -e  # Exit on error

echo "=================================================="
echo "AFlow + ROLL Integration Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_info "Found Python $python_version"

if ! python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)'; then
    print_error "Python 3.10+ required. Found: $python_version"
    exit 1
fi

# Check CUDA availability
print_info "Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    print_info "CUDA is available"
else
    print_warn "CUDA not detected. Training may run on CPU (very slow)."
fi

# Create virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_info "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA if available)
print_info "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch torchvision torchaudio
fi

# Install other dependencies
print_info "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install megatron-core (required for megatron_train strategy)
print_info "Installing megatron-core..."
pip install megatron-core --no-cache-dir

# Install mcore_adapter (required for megatron_train strategy)
print_info "Installing mcore_adapter..."
cd /home/username/ROLL/mcore_adapter
pip install -e . --no-cache-dir
cd /home/username/ROLL/examples/aflow_integration

# Clone and install ROLL if not already installed
if [ ! -d "../../../.git" ]; then
    print_info "Cloning ROLL framework..."
    cd ../../..
    if [ ! -d "ROLL" ]; then
        git clone https://github.com/microsoft/ROLL.git
    fi
    cd ROLL
    print_info "Installing ROLL..."
    pip install -e .
    cd examples/aflow_integration
else
    print_info "ROLL already installed (detected git repository)"
fi

# Clone and install AFlow
print_info "Setting up AFlow framework..."
AFLOW_DIR="/home/username/AFlow"
if [ ! -d "$AFLOW_DIR" ]; then
    print_info "Cloning AFlow framework..."
    cd /home/username
    git clone https://github.com/geekan/AFlow.git
    cd AFlow
    print_info "Installing AFlow..."
    pip install -e .
    cd /home/username/ROLL/examples/aflow_integration
else
    print_info "AFlow already installed at $AFLOW_DIR"
fi

# Download model (if needed)
print_info "Checking Qwen-3-8B model..."
MODEL_PATH="/home/username/.cache/huggingface/models--Qwen--Qwen3-8B"
if [ ! -d "$MODEL_PATH" ]; then
    print_warn "Qwen-3-8B model not found locally"
    print_info "Model will be downloaded automatically during first training run"
    print_info "This may take some time (~16GB download)"
else
    print_info "Qwen-3-8B model found in cache"
fi

# Verify data files
print_info "Verifying data files..."
data_files=(
    "data/math_validate.jsonl"
    "data/gsm8k_validate.jsonl"
    "data/humaneval_validate.jsonl"
    "data/mbpp_validate.jsonl"
    "data/hotpotqa_validate.jsonl"
    "data/drop_validate.jsonl"
)

all_data_present=true
for file in "${data_files[@]}"; do
    if [ -f "$file" ]; then
        count=$(wc -l < "$file")
        print_info "✓ $file ($count samples)"
    else
        print_error "✗ $file (missing)"
        all_data_present=false
    fi
done

if [ "$all_data_present" = false ]; then
    print_error "Some data files are missing!"
    print_info "Please ensure all dataset files are in the data/ directory"
    exit 1
fi

# Create necessary directories
print_info "Creating output directories..."
mkdir -p output/checkpoints
mkdir -p output/logs
mkdir -p /home/username/logs/roll_native_training
print_info "Directories created"

# Set environment variables
print_info "Setting up environment variables..."
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
export PYTHONPATH=/home/username/ROLL:/home/username/AFlow:$PYTHONPATH

# Save environment setup
cat > setup_env.sh << 'EOF'
#!/bin/bash
# Source this file to set up environment variables
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
export PYTHONPATH=/home/username/ROLL:/home/username/AFlow:$PYTHONPATH
source venv/bin/activate
EOF
chmod +x setup_env.sh

print_info "Environment setup script created: setup_env.sh"

# Initialize Ray
print_info "Initializing Ray..."
python3 -c "import ray; ray.init(ignore_reinit_error=True); ray.shutdown()" || true
print_info "Ray initialized"

echo ""
echo "=================================================="
print_info "Setup completed successfully!"
echo "=================================================="
echo ""
print_info "Next steps:"
echo "  1. Activate environment: source setup_env.sh"
echo "  2. Start training: ./run.sh"
echo "  3. Monitor logs: tail -f /home/username/logs/megatron_training.log"
echo ""
print_info "For more information, see README.md"
