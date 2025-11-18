"""
RL Agent Training Module
Trains PPO agent for production scheduling using Ray RLlib
"""

from typing import Dict, Optional
import os
from pathlib import Path
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms import Algorithm
from ray.rllib.env.env_context import EnvContext
import gymnasium as gym

from .environment import ProductionSchedulingEnv
from .reward import RewardConfig


def create_env(env_config: EnvContext) -> gym.Env:
    """Environment creator function for Ray"""
    return ProductionSchedulingEnv(
        num_machines=env_config.get("num_machines", 3),
        max_jobs_per_episode=env_config.get("max_jobs_per_episode", 50),
        episode_duration=env_config.get("episode_duration", 2000),
        reward_config=None,  # Use default
        use_discrete_actions=env_config.get("use_discrete_actions", True),
        seed=env_config.get("seed", None),
    )


class SchedulingRLTrainer:
    """Trainer for RL-based production scheduling"""

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints/rl_scheduling",
        tensorboard_dir: str = "./tensorboard/rl_scheduling",
        num_workers: int = 4,
        num_gpus: int = 0,
    ):
        """
        Initialize trainer

        Args:
            checkpoint_dir: Directory to save model checkpoints
            tensorboard_dir: Directory for tensorboard logs
            num_workers: Number of parallel workers for training
            num_gpus: Number of GPUs to use (0 for CPU only)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.num_workers = num_workers
        self.num_gpus = num_gpus

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                num_cpus=num_workers + 1,
                num_gpus=num_gpus,
                _temp_dir="/tmp/ray",
            )

        # Register environment
        from ray.tune.registry import register_env

        register_env("production_scheduling", create_env)

        self.algorithm: Optional[Algorithm] = None

    def create_config(
        self,
        num_machines: int = 3,
        episode_duration: int = 2000,
        learning_rate: float = 3e-4,
        train_batch_size: int = 4000,
        sgd_minibatch_size: int = 128,
        num_sgd_iter: int = 10,
    ) -> PPOConfig:
        """
        Create PPO algorithm configuration

        Args:
            num_machines: Number of machines in the factory
            episode_duration: Episode duration in minutes
            learning_rate: Learning rate for optimizer
            train_batch_size: Total batch size for training
            sgd_minibatch_size: Minibatch size for SGD
            num_sgd_iter: Number of SGD iterations per training batch

        Returns:
            PPO configuration object
        """
        config = (
            PPOConfig()
            .environment(
                env="production_scheduling",
                env_config={
                    "num_machines": num_machines,
                    "max_jobs_per_episode": 50,
                    "episode_duration": episode_duration,
                    "use_discrete_actions": True,
                    "seed": None,
                },
            )
            .framework("torch")
            .rollouts(
                num_rollout_workers=self.num_workers,
                num_envs_per_worker=1,
                rollout_fragment_length=200,
            )
            .training(
                lr=learning_rate,
                gamma=0.99,
                lambda_=0.95,
                train_batch_size=train_batch_size,
                sgd_minibatch_size=sgd_minibatch_size,
                num_sgd_iter=num_sgd_iter,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                vf_loss_coeff=0.5,
                kl_coeff=0.5,
                model={
                    "fcnet_hiddens": [256, 256, 128],
                    "fcnet_activation": "relu",
                    "vf_share_layers": False,
                },
            )
            .evaluation(
                evaluation_interval=5,
                evaluation_duration=10,
                evaluation_num_workers=1,
                evaluation_config={
                    "explore": False,
                },
            )
            .debugging(
                log_level="INFO",
                seed=42,
            )
            .resources(
                num_gpus=self.num_gpus,
                num_cpus_per_worker=1,
            )
            .reporting(
                min_sample_timesteps_per_iteration=1000,
                min_time_s_per_iteration=10,
            )
        )

        return config

    def train(
        self,
        num_iterations: int = 100,
        checkpoint_freq: int = 10,
        config: Optional[PPOConfig] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict:
        """
        Train the RL agent

        Args:
            num_iterations: Number of training iterations
            checkpoint_freq: Save checkpoint every N iterations
            config: PPO configuration (creates default if None)
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results dictionary
        """
        # Create or use provided config
        if config is None:
            config = self.create_config()

        # Build algorithm
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
            self.algorithm = Algorithm.from_checkpoint(resume_from_checkpoint)
        else:
            print("Starting new training run...")
            self.algorithm = config.build()

        # Training loop
        results_history = []

        for iteration in range(1, num_iterations + 1):
            print(f"\n{'=' * 60}")
            print(f"Training Iteration {iteration}/{num_iterations}")
            print(f"{'=' * 60}")

            # Train one iteration
            result = self.algorithm.train()

            # Extract key metrics
            metrics = {
                "iteration": iteration,
                "episode_reward_mean": result.get("episode_reward_mean", 0),
                "episode_reward_min": result.get("episode_reward_min", 0),
                "episode_reward_max": result.get("episode_reward_max", 0),
                "episode_len_mean": result.get("episode_len_mean", 0),
                "num_env_steps_trained": result.get("num_env_steps_trained", 0),
                "timesteps_total": result.get("timesteps_total", 0),
            }

            # Add custom metrics if available
            if "custom_metrics" in result:
                custom = result["custom_metrics"]
                metrics.update(
                    {
                        "utilization_mean": custom.get("utilization_mean", 0),
                        "num_completed_mean": custom.get("num_completed_mean", 0),
                        "num_overdue_mean": custom.get("num_overdue_mean", 0),
                    }
                )

            results_history.append(metrics)

            # Print metrics
            print(f"\nMetrics:")
            print(f"  Reward (mean): {metrics['episode_reward_mean']:.2f}")
            print(f"  Reward (min/max): {metrics['episode_reward_min']:.2f} / {metrics['episode_reward_max']:.2f}")
            print(f"  Episode length: {metrics['episode_len_mean']:.1f}")
            print(f"  Total timesteps: {metrics['timesteps_total']}")

            # Save checkpoint
            if iteration % checkpoint_freq == 0:
                checkpoint_path = self.algorithm.save(checkpoint_dir=str(self.checkpoint_dir))
                print(f"\n✓ Checkpoint saved: {checkpoint_path}")

        # Final checkpoint
        final_checkpoint = self.algorithm.save(checkpoint_dir=str(self.checkpoint_dir))
        print(f"\n{'=' * 60}")
        print(f"Training Complete!")
        print(f"Final checkpoint: {final_checkpoint}")
        print(f"{'=' * 60}")

        return {
            "final_checkpoint": final_checkpoint,
            "results_history": results_history,
            "final_metrics": results_history[-1] if results_history else {},
        }

    def evaluate(
        self, num_episodes: int = 10, checkpoint_path: Optional[str] = None, render: bool = False
    ) -> Dict:
        """
        Evaluate trained agent

        Args:
            num_episodes: Number of evaluation episodes
            checkpoint_path: Path to checkpoint to load
            render: Whether to render episodes

        Returns:
            Evaluation results
        """
        # Load checkpoint if provided
        if checkpoint_path:
            self.algorithm = Algorithm.from_checkpoint(checkpoint_path)
        elif self.algorithm is None:
            raise ValueError("No algorithm loaded. Provide checkpoint_path or train first.")

        # Create evaluation environment
        env = ProductionSchedulingEnv(
            num_machines=3, episode_duration=2000, use_discrete_actions=True
        )

        episode_rewards = []
        episode_lengths = []
        episode_metrics = []

        print(f"\nEvaluating for {num_episodes} episodes...")

        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                # Get action from policy
                action = self.algorithm.compute_single_action(obs, explore=False)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                if render:
                    env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_metrics.append(info)

            print(
                f"  Episode {episode + 1}: Reward={episode_reward:.2f}, "
                f"Length={episode_length}, Completed={info.get('num_completed', 0)}, "
                f"Overdue={info.get('num_overdue', 0)}"
            )

        env.close()

        # Calculate statistics
        import numpy as np

        results = {
            "num_episodes": num_episodes,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "episode_rewards": episode_rewards,
            "episode_metrics": episode_metrics,
        }

        print(f"\n{'=' * 60}")
        print("Evaluation Results:")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"  Mean Episode Length: {results['mean_length']:.1f}")
        print(f"{'=' * 60}")

        return results

    def export_policy(self, export_dir: str, checkpoint_path: Optional[str] = None):
        """
        Export trained policy for deployment

        Args:
            export_dir: Directory to export policy
            checkpoint_path: Checkpoint to export from
        """
        if checkpoint_path:
            self.algorithm = Algorithm.from_checkpoint(checkpoint_path)

        if self.algorithm is None:
            raise ValueError("No algorithm to export")

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # Export policy weights
        policy = self.algorithm.get_policy()
        policy_state = policy.get_state()

        import pickle

        with open(export_path / "policy_state.pkl", "wb") as f:
            pickle.dump(policy_state, f)

        print(f"Policy exported to: {export_path}")

    def cleanup(self):
        """Cleanup resources"""
        if self.algorithm:
            self.algorithm.stop()

        if ray.is_initialized():
            ray.shutdown()
