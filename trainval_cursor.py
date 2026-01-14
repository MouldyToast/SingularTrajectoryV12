#!/usr/bin/env python3
"""
Cursor Trajectory Training Script

Train SingularTrajectory model on cursor trajectory data.

Usage:
    python trainval_cursor.py --cfg ./config/cursor_medium.json --tag CursorTraj
    python trainval_cursor.py --cfg ./config/cursor_medium.json --tag CursorTraj --test
"""

import os
import argparse
import baseline
from SingularTrajectory import SingularTrajectory
from utils import (
    DotDict,
    get_exp_config,
    print_arguments,
    CursorTransformerDiffusionTrainer
)


def main():
    parser = argparse.ArgumentParser(description="Train cursor trajectory prediction model")
    parser.add_argument('--cfg', default="./config/cursor_medium.json", type=str,
                        help="Config file path")
    parser.add_argument('--tag', default="CursorTrajectory", type=str,
                        help="Tag for the model checkpoint")
    parser.add_argument('--gpu_id', default="0", type=str,
                        help="GPU device ID")
    parser.add_argument('--test', default=False, action='store_true',
                        help="Run in test/evaluation mode")

    args = parser.parse_args()

    print("=" * 60)
    print("Cursor Trajectory Training")
    print("=" * 60)

    print("\n===== Arguments =====")
    print_arguments(vars(args))

    print("\n===== Configs =====")
    hyper_params = get_exp_config(args.cfg)
    print_arguments(hyper_params)

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Get baseline model and hooks
    PredictorModel = getattr(baseline, hyper_params.baseline).TrajectoryPredictor
    hook_func = DotDict({
        "model_forward_pre_hook": getattr(baseline, hyper_params.baseline).model_forward_pre_hook,
        "model_forward": getattr(baseline, hyper_params.baseline).model_forward,
        "model_forward_post_hook": getattr(baseline, hyper_params.baseline).model_forward_post_hook
    })

    # Create trainer
    trainer = CursorTransformerDiffusionTrainer(
        base_model=PredictorModel,
        model=SingularTrajectory,
        hook_func=hook_func,
        args=args,
        hyper_params=hyper_params
    )

    if not args.test:
        # Training mode
        print("\n" + "=" * 60)
        print("Initializing model...")
        print("=" * 60)

        trainer.init_descriptor()

        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)

        trainer.fit()
    else:
        # Test mode
        print("\n" + "=" * 60)
        print("Testing mode")
        print("=" * 60)

        trainer.load_model()
        print("Model loaded, running evaluation...")

        results = trainer.test()

        print("\n" + "=" * 60)
        print(f"Test Results - {hyper_params.dataset}")
        print("=" * 60)
        for metric, value in results.items():
            print(f"  {metric}: {value:.6f}")
        print("=" * 60)


if __name__ == '__main__':
    main()
