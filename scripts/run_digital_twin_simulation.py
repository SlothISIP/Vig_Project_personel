"""Run Digital Twin Factory Simulation.

Demonstrates the digital twin simulation capabilities including:
- IoT sensor simulation
- Production line modeling
- Discrete-event simulation
- What-if analysis
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.digital_twin.simulation import (
    create_factory_simulator,
    create_example_scenarios,
    WhatIfAnalysis,
)
from src.core.logging import get_logger

logger = get_logger(__name__)


def run_basic_simulation(hours: float = 1.0, num_lines: int = 1):
    """
    Run a basic factory simulation.

    Args:
        hours: Simulation duration in hours
        num_lines: Number of production lines
    """
    logger.info("=" * 80)
    logger.info("BASIC FACTORY SIMULATION")
    logger.info("=" * 80)

    # Create simulator
    simulator = create_factory_simulator(
        num_lines=num_lines, simulation_hours=hours
    )

    # Run simulation
    simulator.run()

    # Print statistics
    stats = simulator.get_statistics()

    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print(f"Simulation Duration: {hours} hours")
    print(f"Products Introduced: {stats['total_products_introduced']}")
    print(f"Products Completed: {stats['total_products_completed']}")
    print(f"Products Defective: {stats['total_products_defective']}")
    print(f"Overall Yield: {stats['overall_yield']:.2%}")
    print("=" * 80)


def run_scenario_analysis(hours: float = 2.0):
    """
    Run scenario analysis with predefined scenarios.

    Args:
        hours: Simulation duration in hours
    """
    logger.info("=" * 80)
    logger.info("SCENARIO ANALYSIS")
    logger.info("=" * 80)

    # Create scenario manager with examples
    manager = create_example_scenarios()

    # Factory simulator factory function
    def create_sim():
        return create_factory_simulator(num_lines=1, simulation_hours=hours)

    # Run all scenarios
    results = manager.run_all_scenarios(create_sim)

    # Compare scenarios
    comparison = manager.compare_scenarios(
        metrics=[
            "overall_yield",
            "total_products_completed",
            "total_products_defective",
        ]
    )

    # Print comparison
    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON")
    print("=" * 80)

    for scenario_id, metrics in comparison.items():
        print(f"\n{metrics['name']}:")
        print(f"  Description: {metrics['description']}")
        print(f"  Products Completed: {metrics['total_products_completed']}")
        print(f"  Products Defective: {metrics['total_products_defective']}")
        print(f"  Overall Yield: {metrics['overall_yield']:.2%}")

    # Find best scenario
    best = manager.find_best_scenario("overall_yield", maximize=True)
    if best:
        print(f"\n{'=' * 80}")
        print(f"BEST SCENARIO: {best.name}")
        print(f"Yield: {best.results['overall_yield']:.2%}")
        print(f"Parameters: {best.parameters}")
        print("=" * 80)


def run_parameter_sweep(hours: float = 1.0):
    """
    Run parameter sweep analysis.

    Args:
        hours: Simulation duration in hours
    """
    logger.info("=" * 80)
    logger.info("PARAMETER SWEEP ANALYSIS")
    logger.info("=" * 80)

    # Create What-If analysis
    def create_sim():
        return create_factory_simulator(num_lines=1, simulation_hours=hours)

    analysis = WhatIfAnalysis(create_sim)

    # Sweep product arrival rate
    arrival_rates = [30.0, 45.0, 60.0, 90.0, 120.0]  # seconds
    results = analysis.parameter_sweep("product_arrival_rate", arrival_rates)

    # Print results
    print("\n" + "=" * 80)
    print("PRODUCT ARRIVAL RATE SWEEP")
    print("=" * 80)
    print(f"{'Arrival Rate (s)':<20} {'Completed':<15} {'Defective':<15} {'Yield':<10}")
    print("-" * 80)

    for rate, result in results.items():
        completed = result["total_products_completed"]
        defective = result["total_products_defective"]
        yield_rate = result["overall_yield"]
        print(f"{rate:<20} {completed:<15} {defective:<15} {yield_rate:.2%}")

    print("=" * 80)


def run_sensor_monitoring():
    """Demonstrate sensor monitoring."""
    logger.info("=" * 80)
    logger.info("SENSOR MONITORING DEMO")
    logger.info("=" * 80)

    from src.digital_twin.simulation.sensor import create_standard_sensor_network

    # Create sensor network
    network = create_standard_sensor_network("Demo_Machine_01")

    print("\n" + "=" * 80)
    print("SENSOR NETWORK STATISTICS")
    print("=" * 80)
    stats = network.get_statistics()
    print(f"Network ID: {stats['network_id']}")
    print(f"Total Sensors: {stats['total_sensors']}")
    print(f"Sensors by Type: {stats['sensors_by_type']}")

    # Read sensors 10 times
    print("\n" + "=" * 80)
    print("SENSOR READINGS (10 samples)")
    print("=" * 80)

    for i in range(10):
        readings = network.read_all()
        print(f"\nReading {i+1}:")
        for reading in readings:
            anomaly_flag = "⚠️ ANOMALY" if reading.anomaly_score > 0.5 else ""
            print(
                f"  {reading.sensor_id:<30} "
                f"{reading.value:>8.2f} {reading.unit:<10} "
                f"{anomaly_flag}"
            )

    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run Digital Twin Factory Simulation"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "scenarios", "sweep", "sensors", "all"],
        default="basic",
        help="Simulation mode",
    )

    parser.add_argument(
        "--hours",
        type=float,
        default=1.0,
        help="Simulation duration in hours",
    )

    parser.add_argument(
        "--lines",
        type=int,
        default=1,
        help="Number of production lines",
    )

    args = parser.parse_args()

    try:
        if args.mode == "basic":
            run_basic_simulation(hours=args.hours, num_lines=args.lines)

        elif args.mode == "scenarios":
            run_scenario_analysis(hours=args.hours)

        elif args.mode == "sweep":
            run_parameter_sweep(hours=args.hours)

        elif args.mode == "sensors":
            run_sensor_monitoring()

        elif args.mode == "all":
            run_basic_simulation(hours=args.hours, num_lines=args.lines)
            print("\n\n")
            run_scenario_analysis(hours=args.hours)
            print("\n\n")
            run_parameter_sweep(hours=args.hours)
            print("\n\n")
            run_sensor_monitoring()

    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
