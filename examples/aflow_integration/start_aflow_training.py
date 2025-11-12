"""
Start AFlow + ROLL Training

This script starts RL training for workflow optimization using ROLL's RLVR pipeline.
"""

import argparse
import os
import sys

# Add paths
sys.path.append('/home/username/ROLL')
sys.path.append('/home/username/AFlow')
sys.path.append('/home/username')

from dacite import from_dict, Config
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig
from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="The path of the main configuration file", default=".")
    parser.add_argument(
        "--config_name", help="The name of the main configuration file (without extension).", default="aflow_rlvr_config"
    )
    args = parser.parse_args()

    # Initialize Hydra
    initialize(config_path=args.config_path, job_name="aflow_training")
    cfg = compose(config_name=args.config_name)

    print("=" * 80)
    print("AFlow + ROLL Training - Workflow Optimization with RL")
    print("=" * 80)
    print()
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Convert to RLVRConfig
    rlvr_config: RLVRConfig = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    # Initialize distributed system
    init()

    # Create and run pipeline
    pipeline = RLVRPipeline(pipeline_config=rlvr_config)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nTraining completed")
        print("=" * 80)


if __name__ == "__main__":
    main()
