#!/usr/bin/env python3
"""
Few-Shot Training Script

Train SingularTrajectory model with limited training data to evaluate
few-shot learning capability.

The model's SVD-based representation and anchor clustering provide strong
inductive biases that enable learning from limited examples.

Usage:
    # Train with only 100 samples
    python scripts/few_shot_train.py --cfg ./config/cursor_medium.json \
        --tag FewShot_100 --num_samples 100

    # Train with 10% of data
    python scripts/few_shot_train.py --cfg ./config/cursor_medium.json \
        --tag FewShot_10pct --sample_ratio 0.1
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import baseline
from SingularTrajectory import SingularTrajectory
from utils.cursor_trainer import CursorTransformerDiffusionTrainer
from utils.utils import DotDict, get_exp_config, print_arguments


class FewShotCursorTrainer(CursorTransformerDiffusionTrainer):
    """Cursor trainer with few-shot data subsampling support."""

    def __init__(self, base_model, model, hook_func, args, hyper_params,
                 num_samples=None, sample_ratio=None):
        # Store few-shot parameters before parent init
        self.few_shot_num_samples = num_samples
        self.few_shot_sample_ratio = sample_ratio

        # Call parent init (which creates dataloaders)
        super().__init__(base_model, model, hook_func, args, hyper_params)

        # Apply few-shot subsampling to training data
        self._apply_few_shot_sampling()

    def _apply_few_shot_sampling(self):
        """Subsample training data for few-shot learning."""
        train_dataset = self.loader_train.dataset
        original_size = len(train_dataset)

        if original_size == 0:
            print("Warning: Empty training dataset")
            return

        # Determine target size
        if self.few_shot_num_samples is not None:
            target_size = min(self.few_shot_num_samples, original_size)
        elif self.few_shot_sample_ratio is not None:
            target_size = max(1, int(original_size * self.few_shot_sample_ratio))
        else:
            target_size = original_size

        if target_size >= original_size:
            print(f"Few-shot: Using all {original_size} training samples")
            return

        # Random subsample
        indices = random.sample(range(original_size), target_size)

        # Update dataset to only include subsampled data
        train_dataset.trajectories = [train_dataset.trajectories[i] for i in indices]
        train_dataset.traj_lens = train_dataset.traj_lens[indices]
        train_dataset.non_linear = train_dataset.non_linear[indices]
        train_dataset.scene_ids = train_dataset.scene_ids[indices]
        train_dataset.num_trajectories = target_size
        train_dataset.num_peds_in_seq = np.ones(target_size, dtype=np.int32)
        train_dataset.scene_id = train_dataset.scene_ids

        # Rebuild trajectory tensors
        train_dataset._build_trajectory_tensors()

        # Reset anchor (will be recalculated)
        train_dataset.anchor = None

        print(f"Few-shot: Subsampled to {target_size}/{original_size} training samples "
              f"({100*target_size/original_size:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Few-shot cursor trajectory training")
    parser.add_argument('--cfg', default="./config/cursor_medium.json", type=str,
                        help="Config file path")
    parser.add_argument('--tag', default="FewShot", type=str,
                        help="Tag for the model checkpoint")
    parser.add_argument('--gpu_id', default="0", type=str,
                        help="GPU device ID")
    parser.add_argument('--test', default=False, action='store_true',
                        help="Run in test/evaluation mode")

    # Few-shot parameters
    parser.add_argument('--num_samples', '-n', type=int, default=None,
                        help="Number of training samples to use (absolute)")
    parser.add_argument('--sample_ratio', '-r', type=float, default=None,
                        help="Ratio of training samples to use (0.0-1.0)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for reproducible subsampling")

    args = parser.parse_args()

    # Validate few-shot parameters
    if args.num_samples is None and args.sample_ratio is None:
        print("Warning: Neither --num_samples nor --sample_ratio specified.")
        print("         Using all training data (not few-shot mode)")

    if args.num_samples is not None and args.sample_ratio is not None:
        print("Warning: Both --num_samples and --sample_ratio specified.")
        print("         Using --num_samples, ignoring --sample_ratio")
        args.sample_ratio = None

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("Few-Shot Cursor Trajectory Training")
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

    # Create few-shot trainer
    trainer = FewShotCursorTrainer(
        base_model=PredictorModel,
        model=SingularTrajectory,
        hook_func=hook_func,
        args=args,
        hyper_params=hyper_params,
        num_samples=args.num_samples,
        sample_ratio=args.sample_ratio
    )

    if not args.test:
        # Training mode
        print("\n" + "=" * 60)
        print("Initializing model...")
        print("=" * 60)

        trainer.init_descriptor()

        print("\n" + "=" * 60)
        print("Starting few-shot training...")
        print("=" * 60)

        trainer.fit()

        # Final test
        print("\n" + "=" * 60)
        print("Evaluating few-shot model...")
        print("=" * 60)

        trainer.load_model()
        results = trainer.test()

        print("\n" + "=" * 60)
        print(f"Few-Shot Results - {hyper_params.dataset}")
        print("=" * 60)
        for metric, value in results.items():
            print(f"  {metric}: {value:.6f}")
        print("=" * 60)

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
