#!/usr/bin/env python3
"""
Demo: RL Scheduler in Action
Demonstrates the trained RL agent making scheduling decisions
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl_scheduling import (
    ProductionSchedulingEnv,
    RLSchedulingPolicy,
    DynamicScheduler,
)


def demo_basic_env():
    """Demo: Basic environment simulation"""
    print("\n" + "=" * 70)
    print("Demo 1: Basic Environment Simulation")
    print("=" * 70)

    env = ProductionSchedulingEnv(
        num_machines=3, max_jobs_per_episode=30, episode_duration=1000
    )

    obs, info = env.reset()
    print(f"\n초기 상태:")
    env.render()

    print(f"\nObservation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # Random policy simulation
    print(f"\n랜덤 정책으로 50 스텝 시뮬레이션...")
    total_reward = 0
    steps = 0

    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        steps += 1

        if (i + 1) % 10 == 0:
            print(f"\nStep {i + 1}:")
            env.render()
            print(f"  Reward: {reward:.2f}")
            print(f"  Cumulative: {total_reward:.2f}")

        if terminated or truncated:
            break

    print(f"\n에피소드 완료!")
    print(f"  총 스텝: {steps}")
    print(f"  총 보상: {total_reward:.2f}")
    print(f"  완료된 작업: {info['num_completed']}")
    print(f"  지연된 작업: {info['num_overdue']}")

    env.close()


def demo_trained_policy(checkpoint_path: str):
    """Demo: Trained RL policy"""
    print("\n" + "=" * 70)
    print("Demo 2: Trained RL Policy")
    print("=" * 70)

    # Load policy
    print(f"\n정책 로딩: {checkpoint_path}")
    policy = RLSchedulingPolicy(checkpoint_path, use_ray=True)

    # Create environment
    env = ProductionSchedulingEnv(num_machines=3, episode_duration=1000)

    # Run episodes
    num_episodes = 3
    print(f"\n{num_episodes}개 에피소드 실행...")

    for episode in range(num_episodes):
        print(f"\n{'=' * 70}")
        print(f"Episode {episode + 1}")
        print(f"{'=' * 70}")

        obs, info = env.reset()
        env.render()

        done = False
        episode_reward = 0
        step = 0

        while not done and step < 100:
            # Get action from policy
            action, action_info = policy.predict_action(
                env.state, explore=False
            )

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step += 1

            # Render every 10 steps
            if step % 10 == 0:
                print(f"\nStep {step}:")
                env.render()
                print(f"  Action: {action}")
                print(f"  Reward: {reward:.2f}")
                print(f"  Cumulative Reward: {episode_reward:.2f}")

        print(f"\n에피소드 {episode + 1} 완료!")
        print(f"  총 스텝: {step}")
        print(f"  총 보상: {episode_reward:.2f}")
        print(f"  완료: {info['num_completed']}, 진행중: {info['num_active']}, 지연: {info['num_overdue']}")
        print(f"  평균 가동률: {info['utilization']:.2%}")

    policy.cleanup()
    env.close()


def demo_dynamic_scheduler(checkpoint_path: str):
    """Demo: Dynamic scheduler with RL policy"""
    print("\n" + "=" * 70)
    print("Demo 3: Dynamic Scheduler")
    print("=" * 70)

    # Load policy and create scheduler
    policy = RLSchedulingPolicy(checkpoint_path, use_ray=True)
    scheduler = DynamicScheduler(policy, fallback_enabled=True)

    # Create environment
    env = ProductionSchedulingEnv(num_machines=3, episode_duration=1500)

    obs, info = env.reset()
    print("\n초기 상태:")
    env.render()

    # Run scheduling loop
    print("\n스케줄링 시작...")
    episode_reward = 0
    step = 0
    max_steps = 100

    while step < max_steps:
        # Get scheduling recommendation
        action_obj = scheduler.schedule_next_job(env.state)

        if action_obj:
            # Find corresponding action index
            # Simplified: use first available action
            action = 0  # This should be properly mapped
        else:
            action = env.action_space.sample()

        # Execute
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step += 1

        if step % 20 == 0:
            print(f"\n[Step {step}]")
            env.render()
            print(f"  Reward: {reward:.2f}")
            print(f"  Cumulative: {episode_reward:.2f}")

        if terminated or truncated:
            break

    # Statistics
    print("\n" + "=" * 70)
    print("스케줄러 통계:")
    print("=" * 70)
    stats = scheduler.get_statistics()
    for key, value in stats.items():
        if "rate" in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")

    print(f"\n최종 결과:")
    print(f"  총 보상: {episode_reward:.2f}")
    print(f"  완료: {info['num_completed']}")
    print(f"  지연: {info['num_overdue']}")
    print(f"  가동률: {info['utilization']:.2%}")

    policy.cleanup()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="RL Scheduler Demo")

    parser.add_argument(
        "--demo",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Demo mode: 1=Basic, 2=Trained Policy, 3=Dynamic Scheduler",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained checkpoint (required for demo 2 and 3)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("RL-based Production Scheduling - Demo")
    print("=" * 70)

    try:
        if args.demo == 1:
            demo_basic_env()

        elif args.demo == 2:
            if not args.checkpoint:
                print("Error: --checkpoint required for demo 2")
                return 1
            demo_trained_policy(args.checkpoint)

        elif args.demo == 3:
            if not args.checkpoint:
                print("Error: --checkpoint required for demo 3")
                return 1
            demo_dynamic_scheduler(args.checkpoint)

        print("\n데모 완료!")
        return 0

    except KeyboardInterrupt:
        print("\n\n데모 중단")
        return 1

    except Exception as e:
        print(f"\n에러 발생: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
