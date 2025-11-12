# AFlow + ROLL Integration

**One-Click Multi-Domain Reinforcement Learning for LLMs**

Train Qwen-3-8B across 6 reasoning domains using AFlow workflow optimization and ROLL's efficient RL framework.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🚀 Quick Start (One-Click Setup)

```bash
# 1. Clone the repository
git clone https://github.com/beita6969/icml.git
cd icml

# 2. Run automated setup (installs all dependencies)
chmod +x setup.sh run.sh
./setup.sh

# 3. Start training with one command
./run.sh
```

That's it! The training will start automatically and run in the background.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

## 🎯 Overview

This project integrates two powerful frameworks:
- **AFlow**: Workflow optimization for complex reasoning tasks
- **ROLL**: Efficient reinforcement learning for large language models

### Training Domains

Train across 6 diverse reasoning domains:

| Domain | Description | Samples | Weight |
|--------|-------------|---------|--------|
| **MATH** | Mathematical problem solving | 271 | 30% |
| **GSM8K** | Grade school math | 180 | 20% |
| **HumanEval** | Python code generation | 180 | 20% |
| **MBPP** | Multi-language code | 135 | 15% |
| **HotpotQA** | Multi-hop QA | 90 | 10% |
| **DROP** | Discrete reasoning | 46 | 5% |

**Total**: 902 training samples, 119 validation samples

## ✨ Features

- **One-Click Setup**: Automated installation of all dependencies
- **One-Click Training**: Start training with a single command
- **Multi-Domain**: Train across 6 reasoning domains simultaneously
- **GRPO Algorithm**: State-of-the-art Group Relative Policy Optimization
- **Single GPU Optimized**: Efficient training on 1x A100 (or similar)
- **ROLL-Native**: Uses GeneralValRuleRewardWorker (no external dependencies)
- **Comprehensive Logging**: Detailed training logs and metrics
- **Checkpointing**: Automatic checkpoint saving every 100 steps

## 📦 Installation

### Prerequisites

- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with 40GB+ VRAM (e.g., A100, A6000)
- **CUDA**: 11.8 or higher
- **Disk**: 100GB+ free space

### Automated Setup

The `setup.sh` script automatically:
1. Checks Python and CUDA versions
2. Creates virtual environment
3. Installs PyTorch with CUDA support
4. Installs all Python dependencies
5. Clones and installs ROLL framework
6. Clones and installs AFlow framework
7. Verifies data files
8. Creates necessary directories
9. Initializes Ray cluster

```bash
./setup.sh
```

### Manual Installation

If you prefer manual installation:

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt

# 4. Clone and install ROLL
git clone https://github.com/microsoft/ROLL.git
cd ROLL && pip install -e . && cd ..

# 5. Clone and install AFlow
git clone https://github.com/geekan/AFlow.git
cd AFlow && pip install -e . && cd ..

# 6. Set environment variables
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
export PYTHONPATH=/path/to/ROLL:/path/to/AFlow:$PYTHONPATH
```

## 🎮 Usage

### One-Click Training

```bash
./run.sh
```

This script will:
1. Load the environment
2. Check GPU availability
3. Stop any existing Ray instances
4. Start the training process in background
5. Show you how to monitor progress

### Manual Training

```bash
# Source environment
source setup_env.sh

# Start training
python start_roll_native_training.py
```

### Monitoring Training

#### Real-time Log Monitoring

```bash
# Follow the latest log file
tail -f /home/username/logs/roll_native_training/training_*.log

# Watch last 20 lines with auto-refresh
watch -n 5 'tail -20 /home/username/logs/roll_native_training/training_*.log'
```

#### Check Training Progress

```bash
# See training steps, loss, and rewards
grep -E '(step|loss|reward)' /home/username/logs/roll_native_training/training_*.log | tail -20

# Check worker status
ps aux | grep start_roll_native_training
```

#### TensorBoard (if configured)

```bash
tensorboard --logdir=/home/username/logs/aflow_training/tensorboard
```

### Stopping Training

```bash
# Find the process ID
ps aux | grep start_roll_native_training

# Kill the process
kill <PID>

# Stop Ray
ray stop
```

## ⚙️ Configuration

### Main Configuration File

Edit `roll_native_config.yaml` to customize training:

```yaml
# Training schedule
max_steps: 1000              # Total training steps
save_steps: 100              # Checkpoint frequency
eval_steps: 50               # Validation frequency

# GRPO settings
rollout_batch_size: 16       # Prompts per batch
num_return_sequences_in_group: 4  # Samples per prompt
kl_loss_coef: 0.001          # KL divergence weight

# Model paths
pretrain: /path/to/qwen-3-8b
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_steps` | 1000 | Total training iterations |
| `rollout_batch_size` | 16 | Batch size for generation |
| `learning_rate` | 1e-6 | Actor learning rate |
| `sequence_length` | 3072 | Max sequence length (prompt + response) |
| `response_length` | 2048 | Max response length |

### Domain Sampling

Adjust domain weights in `roll_native_config.yaml`:

```yaml
domain_interleave_probs:
  math: 0.3      # 30%
  gsm8k: 0.2     # 20%
  code_human: 0.2     # 20%
  code_mbpp: 0.15    # 15%
  qa_hotpot: 0.1     # 10%
  qa_drop: 0.05      # 5%
```

## 📊 Monitoring

### Training Metrics

The training logs show:
- **Step**: Current training iteration
- **Loss**: Actor loss value
- **Reward**: Average reward across domains
- **KL**: KL divergence from reference model
- **GPU Memory**: Memory usage

### Example Log Output

```
[2024-11-12 10:30:15] Step 100/1000
[2024-11-12 10:30:15] Actor Loss: 0.234
[2024-11-12 10:30:15] Average Reward: 0.567
[2024-11-12 10:30:15] KL Divergence: 0.012
[2024-11-12 10:30:15] GPU Memory: 45.2 GB / 80.0 GB
```

### Checkpoints

Checkpoints are saved to:
```
/home/username/checkpoints/aflow-roll-native-qwen-8b-grpo/
├── step_100/
├── step_200/
├── step_300/
...
```

Load a checkpoint:
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/home/username/checkpoints/aflow-roll-native-qwen-8b-grpo/step_1000"
)
```

## 📁 Project Structure

```
icml/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── setup.sh                       # Automated setup script
├── run.sh                         # One-click training script
├── .gitignore                     # Git ignore rules
│
├── roll_native_config.yaml        # Main training configuration
├── start_roll_native_training.py  # Training entry point
├── convert_aflow_to_roll.py       # Data conversion script
│
├── data/                          # Training and validation data
│   ├── math_validate.jsonl        # MATH dataset (271 samples)
│   ├── gsm8k_validate.jsonl       # GSM8K dataset (180 samples)
│   ├── humaneval_validate.jsonl   # HumanEval dataset (180 samples)
│   ├── mbpp_validate.jsonl        # MBPP dataset (135 samples)
│   ├── hotpotqa_validate.jsonl    # HotpotQA dataset (90 samples)
│   ├── drop_validate.jsonl        # DROP dataset (46 samples)
│   └── all_val_samples.jsonl      # All validation samples (119)
│
└── output/                        # Training outputs
    ├── checkpoints/               # Model checkpoints
    └── logs/                      # Training logs
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Problem**: GPU runs out of memory during training

**Solution**:
```yaml
# In roll_native_config.yaml, reduce:
rollout_batch_size: 8              # Was: 16
response_length: 1024              # Was: 2048
```

#### 2. Model Download Fails

**Problem**: Qwen-3-8B model fails to download

**Solution**:
```bash
# Manually download the model
huggingface-cli download Qwen/Qwen3-8B --local-dir /home/username/.cache/huggingface/models--Qwen--Qwen3-8B

# Or set your HuggingFace token
export HF_TOKEN=your_token_here
```

#### 3. Ray Connection Issues

**Problem**: Ray fails to initialize

**Solution**:
```bash
# Stop all Ray processes
ray stop

# Restart Ray
ray start --head

# Try training again
./run.sh
```

#### 4. Import Errors

**Problem**: Cannot import ROLL or AFlow

**Solution**:
```bash
# Re-run setup
./setup.sh

# Or manually set PYTHONPATH
source setup_env.sh
```

#### 5. Training Process Dies

**Problem**: Training stops unexpectedly

**Solution**:
```bash
# Check logs for errors
tail -100 /home/username/logs/roll_native_training/training_*.log

# Check system resources
nvidia-smi
htop

# Resume from checkpoint (if available)
# Edit roll_native_config.yaml:
resume_from_checkpoint: true
```

### Getting Help

If you encounter issues:
1. Check the logs in `/home/username/logs/roll_native_training/`
2. Open an issue on [GitHub](https://github.com/beita6969/icml/issues)
3. Include:
   - Error message
   - Training configuration
   - System info (GPU, CUDA version, Python version)

## 🔧 Advanced Usage

### Custom Domain

Add your own domain:

1. Prepare data in JSONL format:
```json
{
  "id": "custom_001",
  "messages": [{"role": "user", "content": "Your prompt"}],
  "ground_truth": "Expected answer",
  "tag": "custom_domain"
}
```

2. Update `roll_native_config.yaml`:
```yaml
actor_train:
  data_args:
    file_name:
      - data/custom_domain.jsonl
    domain_interleave_probs:
      custom_domain: 0.1

rewards:
  custom_domain:
    worker_cls: roll.pipeline.rlvr.rewards.general_val_rule_reward_worker.GeneralValRuleRewardWorker
    tag_included: [custom_domain]
```

### Distributed Training

For multi-GPU training:

```yaml
# In roll_native_config.yaml
num_gpus_per_node: 4  # Number of GPUs

actor_train:
  strategy_args:
    strategy_config:
      tensor_model_parallel_size: 2   # Tensor parallelism
      pipeline_model_parallel_size: 2  # Pipeline parallelism
```

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@software{aflow_roll_integration_2024,
  title={AFlow + ROLL Integration: Multi-Domain RL Training for LLMs},
  author={Your Name},
  year={2024},
  url={https://github.com/beita6969/icml},
  note={One-click reinforcement learning training across 6 reasoning domains}
}
```

Also cite the original frameworks:

```bibtex
@software{roll2024,
  title={ROLL: Reinforcement Learning Optimization for Large-Scale Learning},
  author={Microsoft},
  url={https://github.com/microsoft/ROLL},
  year={2024}
}

@software{aflow2024,
  title={AFlow: Workflow Optimization Framework},
  author={AFlow Team},
  url={https://github.com/geekan/AFlow},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ROLL](https://github.com/microsoft/ROLL) by Microsoft Research
- [AFlow](https://github.com/geekan/AFlow) for workflow optimization
- [Qwen](https://github.com/QwenLM/Qwen) team for the base model
- [HuggingFace](https://huggingface.co/) for the transformers library

## 📞 Contact

- **GitHub Issues**: [https://github.com/beita6969/icml/issues](https://github.com/beita6969/icml/issues)
- **Repository**: [https://github.com/beita6969/icml](https://github.com/beita6969/icml)

---

**Happy Training!** 🚀

For questions or support, please open an issue on GitHub.
