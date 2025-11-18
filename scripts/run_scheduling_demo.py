"""Production Scheduling Demo.

Demonstrates OR-Tools based job shop scheduling.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scheduling.models import (
    Job,
    Task,
    Machine,
    create_sample_job,
    create_sample_machines,
)
from src.scheduling.solvers.job_shop_solver import JobShopSolver, SolverConfig
from src.scheduling.optimizer import ScheduleOptimizer, OptimizationObjective
from src.scheduling.scheduler import ProductionScheduler
from src.core.logging import get_logger

logger = get_logger(__name__)


def demo_basic_scheduling():
    """Demonstrate basic job shop scheduling."""
    print("=" * 80)
    print("BASIC JOB SHOP SCHEDULING DEMO")
    print("=" * 80)

    # Create jobs
    jobs = [
        create_sample_job("Job1", num_tasks=3, task_duration=20),
        create_sample_job("Job2", num_tasks=4, task_duration=25),
        create_sample_job("Job3", num_tasks=3, task_duration=30),
    ]

    # Create machines
    machines = create_sample_machines(num_machines=2)

    # Create solver
    config = SolverConfig(max_time_seconds=60, log_search_progress=False)
    solver = JobShopSolver(config=config)

    # Solve
    print(f"\nSolving schedule for {len(jobs)} jobs on {len(machines)} machines...")
    schedule = solver.solve(jobs, machines)

    if schedule:
        # Print statistics
        stats = schedule.get_statistics()

        print("\n" + "=" * 80)
        print("SCHEDULE STATISTICS")
        print("=" * 80)
        print(f"Schedule ID: {stats['schedule_id']}")
        print(f"Total Assignments: {stats['total_assignments']}")
        print(f"Makespan: {stats['makespan_minutes']} minutes")
        print(f"Average Utilization: {stats['average_utilization']:.2%}")
        print(f"Has Conflicts: {stats['has_conflicts']}")
        print(f"Solver Time: {stats['solver_time_seconds']:.2f} seconds")

        # Print assignments
        print("\n" + "=" * 80)
        print("SCHEDULE ASSIGNMENTS")
        print("=" * 80)
        print(f"{'Job':<10} {'Task':<15} {'Machine':<10} {'Start':<20} {'Duration':<10}")
        print("-" * 80)

        for assignment in sorted(
            schedule.assignments, key=lambda a: a.start_time
        ):
            print(
                f"{assignment.job_id:<10} {assignment.task_id:<15} "
                f"{assignment.machine_id:<10} "
                f"{assignment.start_time.strftime('%H:%M:%S'):<20} "
                f"{assignment.get_duration_minutes():<10} min"
            )

        print("=" * 80)
    else:
        print("\nFailed to find a feasible schedule")


def demo_production_scheduler():
    """Demonstrate high-level production scheduler."""
    print("\n\n")
    print("=" * 80)
    print("PRODUCTION SCHEDULER DEMO")
    print("=" * 80)

    # Create machines
    machines = create_sample_machines(num_machines=3)

    # Create scheduler
    config = SolverConfig(max_time_seconds=60)
    scheduler = ProductionScheduler(machines=machines, config=config)

    # Initial jobs
    initial_jobs = [
        create_sample_job("InitialJob1", num_tasks=4, task_duration=20),
        create_sample_job("InitialJob2", num_tasks=3, task_duration=25),
    ]

    print(f"\nScheduling {len(initial_jobs)} initial jobs...")
    result = scheduler.schedule_jobs(initial_jobs)

    if result.success:
        print("\n✓ Initial schedule created")
        print(f"Makespan: {result.metrics.get('makespan', 0)} minutes")
        print(f"Utilization: {result.metrics.get('average_utilization', 0):.2%}")

        # Add new jobs
        print("\nAdding 2 new urgent jobs...")
        new_jobs = [
            create_sample_job("UrgentJob1", num_tasks=2, task_duration=15),
            create_sample_job("UrgentJob2", num_tasks=3, task_duration=20),
        ]

        result2 = scheduler.add_new_jobs(new_jobs)

        if result2.success:
            print("\n✓ Schedule updated with new jobs")
            print(f"New Makespan: {result2.metrics.get('makespan', 0)} minutes")
            print(f"New Utilization: {result2.metrics.get('average_utilization', 0):.2%}")

        # Machine status
        print("\n" + "=" * 80)
        print("MACHINE STATUS")
        print("=" * 80)

        for machine in machines:
            status = scheduler.get_machine_status(machine.machine_id)
            print(f"\n{machine.machine_id}:")
            print(f"  Type: {status['machine_type']}")
            print(f"  Availability: {status['availability']}")
            if "utilization" in status:
                print(f"  Utilization: {status['utilization']:.2%}")
            if "scheduled_tasks" in status:
                print(f"  Scheduled Tasks: {status['scheduled_tasks']}")

        print("=" * 80)


def demo_optimization():
    """Demonstrate schedule optimization."""
    print("\n\n")
    print("=" * 80)
    print("SCHEDULE OPTIMIZATION DEMO")
    print("=" * 80)

    # Create complex jobs
    jobs = [
        create_sample_job("ComplexJob1", num_tasks=5, task_duration=30),
        create_sample_job("ComplexJob2", num_tasks=4, task_duration=25),
        create_sample_job("ComplexJob3", num_tasks=6, task_duration=20),
        create_sample_job("ComplexJob4", num_tasks=3, task_duration=35),
    ]

    machines = create_sample_machines(num_machines=4)

    # Create optimizer
    config = SolverConfig(max_time_seconds=120)
    optimizer = ScheduleOptimizer(solver_config=config)

    # Optimize with makespan objective
    print("\nOptimizing for minimum makespan...")
    schedule = optimizer.optimize(
        jobs, machines, objective=OptimizationObjective.MINIMIZE_MAKESPAN
    )

    if schedule:
        metrics = optimizer.evaluate_schedule(schedule, machines)

        print("\n" + "=" * 80)
        print("OPTIMIZATION RESULTS")
        print("=" * 80)
        print(f"Makespan: {metrics['makespan']} minutes")
        print(f"Average Utilization: {metrics['average_utilization']:.2%}")
        print(f"Total Assignments: {metrics['total_assignments']}")

        print("\nMachine Utilization:")
        for machine in machines:
            util_key = f"utilization_{machine.machine_id}"
            if util_key in metrics:
                print(f"  {machine.machine_id}: {metrics[util_key]:.2%}")

        print("=" * 80)


def demo_custom_problem():
    """Demonstrate scheduling with custom problem."""
    print("\n\n")
    print("=" * 80)
    print("CUSTOM SCHEDULING PROBLEM")
    print("=" * 80)

    # Create machines with different capabilities
    machines = [
        Machine(
            machine_id="Cutter1",
            machine_type="cutting",
            capabilities=["cut", "trim"],
            speed_factor=1.2,
        ),
        Machine(
            machine_id="Welder1",
            machine_type="welding",
            capabilities=["weld"],
            speed_factor=1.0,
        ),
        Machine(
            machine_id="Assembler1",
            machine_type="assembly",
            capabilities=["assemble", "inspect"],
            speed_factor=0.9,
        ),
    ]

    # Create job with specific task sequence
    job = Job(job_id="CustomProduct", product_type="Widget", priority=1)

    # Task sequence: Cut -> Weld -> Assemble -> Inspect
    tasks = [
        Task(task_id="Cut", task_type="cut", duration=20, required_capability="cut"),
        Task(
            task_id="Weld",
            task_type="weld",
            duration=30,
            required_capability="weld",
            predecessor_task_ids=["Cut"],
        ),
        Task(
            task_id="Assemble",
            task_type="assemble",
            duration=40,
            required_capability="assemble",
            predecessor_task_ids=["Weld"],
        ),
        Task(
            task_id="Inspect",
            task_type="inspect",
            duration=15,
            required_capability="inspect",
            predecessor_task_ids=["Assemble"],
        ),
    ]

    for task in tasks:
        job.add_task(task)

    print(f"\nJob: {job.job_id}")
    print(f"Tasks: {len(job.tasks)}")
    print(f"Total Duration: {job.get_total_duration()} minutes")
    print(f"Critical Path: {job.get_critical_path_length()} minutes")

    # Schedule
    solver = JobShopSolver(config=SolverConfig(max_time_seconds=60))
    schedule = solver.solve([job], machines)

    if schedule:
        print("\n" + "=" * 80)
        print("CUSTOM SCHEDULE")
        print("=" * 80)

        for assignment in schedule.assignments:
            print(
                f"{assignment.task_id:<15} -> {assignment.machine_id:<15} "
                f"[{assignment.start_time.strftime('%H:%M')} - "
                f"{assignment.end_time.strftime('%H:%M')}]"
            )

        print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Production Scheduling Demo")

    parser.add_argument(
        "--demo",
        type=str,
        choices=["basic", "scheduler", "optimization", "custom", "all"],
        default="all",
        help="Demo to run",
    )

    args = parser.parse_args()

    try:
        if args.demo == "basic":
            demo_basic_scheduling()
        elif args.demo == "scheduler":
            demo_production_scheduler()
        elif args.demo == "optimization":
            demo_optimization()
        elif args.demo == "custom":
            demo_custom_problem()
        elif args.demo == "all":
            demo_basic_scheduling()
            demo_production_scheduler()
            demo_optimization()
            demo_custom_problem()

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
