#!/usr/bin/env python
"""
ROLL Native Training Script for Multi-Domain Learning

Uses ROLL's built-in reward workers instead of AFlow benchmarks.
Maintains same data (902 AFlow samples) and training setup.

Key differences from aflow_reward_worker integration:
- Uses GeneralValRuleRewardWorker for all domains
- No dependency on AFlow benchmark classes
- Faster to start and more stable
"""

import os
import sys
from pathlib import Path

# Add ROLL to path
roll_path = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, roll_path)

# Set PYTHONPATH for Ray workers
if 'PYTHONPATH' in os.environ:
    os.environ['PYTHONPATH'] = f"{roll_path}:{os.environ['PYTHONPATH']}"
else:
    os.environ['PYTHONPATH'] = roll_path

from dacite import from_dict, Config
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline


def main():
    """Launch ROLL-native multi-domain training"""

    print("=" * 80)
    print("ROLL Native Multi-Domain Training")
    print("=" * 80)
    print()
    print("Configuration:")
    print("  - Model: Qwen3-8B (from local cache)")
    print("  - Algorithm: GRPO")
    print("  - Datasets: 6 domains (MATH, GSM8K, HumanEval, MBPP, HotpotQA, DROP)")
    print("  - Training samples: 902")
    print("  - Validation samples: 119")
    print("  - Max steps: 1000")
    print("  - Reward workers: GeneralValRuleRewardWorker (ROLL native)")
    print()
    print("=" * 80)
    print()

    # Load configuration
    config_path = Path(__file__).parent / "roll_native_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading configuration from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Verify data files exist
    data_dir = Path(__file__).parent / "data"
    required_files = [
        "math_validate.jsonl",
        "gsm8k_validate.jsonl",
        "humaneval_validate.jsonl",
        "mbpp_validate.jsonl",
        "hotpotqa_validate.jsonl",
        "drop_validate.jsonl",
        "all_val_samples.jsonl"
    ]

    print("\nVerifying data files:")
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} - NOT FOUND!")
            raise FileNotFoundError(f"Required data file missing: {filepath}")

    print("\nConverting configuration to RLVRConfig dataclass...")
    rlvr_config: RLVRConfig = from_dict(
        data_class=RLVRConfig,
        data=OmegaConf.to_container(cfg, resolve=True)
    )

    print("\nInitializing distributed environment (Ray)...")
    init()

    print("\nInitializing RLVR pipeline...")
    pipeline = RLVRPipeline(pipeline_config=rlvr_config)

    print("\nStarting training...")
    print("=" * 80)
    pipeline.run()

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
