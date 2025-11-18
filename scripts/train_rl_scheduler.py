#!/usr/bin/env python3
"""
Train RL-based Production Scheduler
Trains a PPO agent to learn optimal scheduling policies
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_scheduling import SchedulingRLTrainer, RewardConfig


def main():
    parser = argparse.ArgumentParser(
        description="Train RL agent for production scheduling"
    )

    # Training parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of training iterations (default: 100)",
    )
    parser.add_argument(
        "--num-machines",
        type=int,
        default=3,
        help="Number of machines in factory (default: 3)",
    )
    parser.add_argument(
        "--episode-duration",
        type=int,
        default=2000,
        help="Episode duration in minutes (default: 2000)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of GPUs to use (default: 0)"
    )

    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints/rl_scheduling",
        help="Checkpoint directory (default: ./checkpoints/rl_scheduling)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10,
        help="Save checkpoint every N iterations (default: 10)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path (default: None)",
    )

    # Hyperparameters
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4000,
        help="Train batch size (default: 4000)",
    )
    parser.add_argument(
        "--sgd-minibatch",
        type=int,
        default=128,
        help="SGD minibatch size (default: 128)",
    )

    # Modes
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate existing checkpoint",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RL-based Production Scheduling - Training")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Machines: {args.num_machines}")
    print(f"  Episode Duration: {args.episode_duration} min")
    print(f"  Workers: {args.workers}")
    print(f"  GPUs: {args.gpus}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Checkpoint Dir: {args.checkpoint_dir}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print()

    # Initialize trainer
    trainer = SchedulingRLTrainer(
        checkpoint_dir=args.checkpoint_dir,
        tensorboard_dir="./tensorboard/rl_scheduling",
        num_workers=args.workers,
        num_gpus=args.gpus,
    )

    try:
        if args.evaluate_only:
            # Evaluation mode
            if not args.resume:
                print("Error: --resume checkpoint required for evaluation")
                return 1

            print(f"\nEvaluating checkpoint: {args.resume}")
            results = trainer.evaluate(
                num_episodes=args.num_eval_episodes,
                checkpoint_path=args.resume,
                render=False,
            )

            print("\nEvaluation complete!")
            return 0

        # Training mode
        config = trainer.create_config(
            num_machines=args.num_machines,
            episode_duration=args.episode_duration,
            learning_rate=args.lr,
            train_batch_size=args.batch_size,
            sgd_minibatch_size=args.sgd_minibatch,
        )

        results = trainer.train(
            num_iterations=args.iterations,
            checkpoint_freq=args.checkpoint_freq,
            config=config,
            resume_from_checkpoint=args.resume,
        )

        print("\n" + "=" * 70)
        print("Training Summary:")
        print("=" * 70)
        print(f"Final Checkpoint: {results['final_checkpoint']}")
        print(f"Total Iterations: {len(results['results_history'])}")

        if results["results_history"]:
            final_metrics = results["results_history"][-1]
            print(f"\nFinal Performance:")
            print(f"  Mean Reward: {final_metrics.get('episode_reward_mean', 0):.2f}")
            print(f"  Episode Length: {final_metrics.get('episode_len_mean', 0):.1f}")
            print(f"  Total Timesteps: {final_metrics.get('timesteps_total', 0)}")

        # Run evaluation
        print("\n" + "=" * 70)
        print("Running Final Evaluation...")
        print("=" * 70)

        eval_results = trainer.evaluate(
            num_episodes=args.num_eval_episodes,
            checkpoint_path=results["final_checkpoint"],
            render=False,
        )

        print("\nTraining complete!")
        print(f"\nCheckpoint saved to: {results['final_checkpoint']}")
        print(
            f"\nTo evaluate later, run:\n  python {sys.argv[0]} --evaluate-only --resume {results['final_checkpoint']}"
        )

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        return 1

    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        trainer.cleanup()


if __name__ == "__main__":
    sys.exit(main())
