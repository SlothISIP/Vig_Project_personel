"""Scenario Management and What-If Analysis.

Tools for defining and running simulation scenarios to analyze factory performance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from copy import deepcopy
import json

from src.digital_twin.simulation.simulator import FactorySimulator, SimulationConfig
from src.digital_twin.simulation.production_line import ProductionLine, WorkStation
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScenarioParameter:
    """A parameter that can be varied in a scenario."""

    name: str
    description: str
    base_value: Any
    test_values: List[Any]
    parameter_type: str  # "station", "line", "simulation"
    target_id: Optional[str] = None  # ID of station/line to modify


@dataclass
class Scenario:
    """
    A simulation scenario with specific parameters.

    Used for What-If analysis by varying parameters and comparing results.
    """

    scenario_id: str
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "results": self.results,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """Create scenario from dictionary."""
        return cls(
            scenario_id=data["scenario_id"],
            name=data["name"],
            description=data["description"],
            parameters=data.get("parameters", {}),
            results=data.get("results"),
        )


class ScenarioManager:
    """
    Manages simulation scenarios and runs What-If analysis.

    Allows users to:
    - Define scenarios with varied parameters
    - Run simulations for each scenario
    - Compare results across scenarios
    - Identify optimal configurations
    """

    def __init__(self):
        """Initialize scenario manager."""
        self.scenarios: Dict[str, Scenario] = {}
        self.baseline_scenario: Optional[Scenario] = None

    def create_scenario(
        self,
        scenario_id: str,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ) -> Scenario:
        """
        Create a new scenario.

        Args:
            scenario_id: Unique identifier
            name: Scenario name
            description: Scenario description
            parameters: Dictionary of parameter overrides

        Returns:
            Created scenario
        """
        scenario = Scenario(
            scenario_id=scenario_id,
            name=name,
            description=description,
            parameters=parameters,
        )

        self.scenarios[scenario_id] = scenario
        logger.info(f"Scenario created: {name} ({scenario_id})")

        return scenario

    def set_baseline(self, scenario: Scenario) -> None:
        """
        Set baseline scenario for comparison.

        Args:
            scenario: Baseline scenario
        """
        self.baseline_scenario = scenario
        logger.info(f"Baseline scenario set: {scenario.name}")

    def apply_scenario_to_simulator(
        self, scenario: Scenario, simulator: FactorySimulator
    ) -> None:
        """
        Apply scenario parameters to simulator.

        Args:
            scenario: Scenario to apply
            simulator: Factory simulator to modify
        """
        params = scenario.parameters

        # Apply simulation parameters
        if "product_arrival_rate" in params:
            simulator.config.product_arrival_rate = params["product_arrival_rate"]

        if "simulation_duration" in params:
            simulator.config.duration = params["simulation_duration"]

        # Apply station parameters
        for key, value in params.items():
            if key.startswith("station_"):
                # Format: station_{line_id}_{station_id}_{parameter}
                parts = key.split("_", 3)
                if len(parts) == 4:
                    _, line_id, station_id, param_name = parts
                    line = simulator.production_lines.get(line_id)
                    if line:
                        station = line.get_station(station_id)
                        if station and hasattr(station, param_name):
                            setattr(station, param_name, value)
                            logger.debug(
                                f"Applied {param_name}={value} to {station_id}"
                            )

        logger.info(f"Scenario {scenario.name} applied to simulator")

    def run_scenario(
        self,
        scenario: Scenario,
        simulator_factory: Callable[[], FactorySimulator],
    ) -> Dict[str, Any]:
        """
        Run a scenario simulation.

        Args:
            scenario: Scenario to run
            simulator_factory: Function that creates a fresh simulator

        Returns:
            Simulation results
        """
        logger.info(f"Running scenario: {scenario.name}")

        # Create fresh simulator
        simulator = simulator_factory()

        # Apply scenario
        self.apply_scenario_to_simulator(scenario, simulator)

        # Run simulation
        simulator.run()

        # Get results
        results = simulator.get_statistics()

        # Store results in scenario
        scenario.results = results

        logger.info(f"Scenario {scenario.name} completed")

        return results

    def run_all_scenarios(
        self, simulator_factory: Callable[[], FactorySimulator]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run all defined scenarios.

        Args:
            simulator_factory: Function that creates a fresh simulator

        Returns:
            Dictionary of scenario_id -> results
        """
        results = {}

        for scenario_id, scenario in self.scenarios.items():
            scenario_results = self.run_scenario(scenario, simulator_factory)
            results[scenario_id] = scenario_results

        logger.info(f"All {len(self.scenarios)} scenarios completed")

        return results

    def compare_scenarios(
        self, metrics: List[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare scenarios on specified metrics.

        Args:
            metrics: List of metric names to compare
                    (e.g., ["overall_yield", "total_products_completed"])

        Returns:
            Comparison data
        """
        if metrics is None:
            metrics = [
                "overall_yield",
                "total_products_completed",
                "total_products_defective",
            ]

        comparison = {}

        for scenario_id, scenario in self.scenarios.items():
            if scenario.results is None:
                logger.warning(f"Scenario {scenario_id} has no results")
                continue

            scenario_metrics = {"name": scenario.name, "description": scenario.description}

            for metric in metrics:
                # Handle nested metrics (e.g., "production_lines.Line_01.total_completed")
                value = self._get_nested_value(scenario.results, metric)
                scenario_metrics[metric] = value

            comparison[scenario_id] = scenario_metrics

        return comparison

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """Get nested dictionary value using dot notation."""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def find_best_scenario(self, metric: str, maximize: bool = True) -> Optional[Scenario]:
        """
        Find the best scenario based on a metric.

        Args:
            metric: Metric to optimize
            maximize: True to maximize, False to minimize

        Returns:
            Best scenario, or None if no scenarios have results
        """
        best_scenario = None
        best_value = None

        for scenario in self.scenarios.values():
            if scenario.results is None:
                continue

            value = self._get_nested_value(scenario.results, metric)

            if value is None:
                continue

            if best_value is None:
                best_scenario = scenario
                best_value = value
            elif maximize and value > best_value:
                best_scenario = scenario
                best_value = value
            elif not maximize and value < best_value:
                best_scenario = scenario
                best_value = value

        if best_scenario:
            logger.info(
                f"Best scenario for {metric}: {best_scenario.name} "
                f"({metric}={best_value})"
            )

        return best_scenario

    def export_comparison(self, filepath: str) -> None:
        """
        Export scenario comparison to JSON file.

        Args:
            filepath: Path to save comparison
        """
        comparison = self.compare_scenarios()

        with open(filepath, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Scenario comparison exported to {filepath}")

    def clear_scenarios(self) -> None:
        """Clear all scenarios."""
        self.scenarios.clear()
        self.baseline_scenario = None
        logger.info("All scenarios cleared")


class WhatIfAnalysis:
    """
    Helper class for conducting What-If analysis.

    Simplifies common What-If analysis patterns:
    - Parameter sweeps
    - Multi-parameter optimization
    - Sensitivity analysis
    """

    def __init__(self, simulator_factory: Callable[[], FactorySimulator]):
        """
        Initialize What-If analysis.

        Args:
            simulator_factory: Function that creates a fresh simulator
        """
        self.simulator_factory = simulator_factory
        self.scenario_manager = ScenarioManager()

    def parameter_sweep(
        self,
        parameter_name: str,
        values: List[Any],
        base_parameters: Dict[str, Any] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform parameter sweep: vary one parameter across multiple values.

        Args:
            parameter_name: Name of parameter to vary
            values: List of values to test
            base_parameters: Base parameter values

        Returns:
            Results for each parameter value
        """
        if base_parameters is None:
            base_parameters = {}

        results = {}

        for i, value in enumerate(values):
            # Create scenario
            params = base_parameters.copy()
            params[parameter_name] = value

            scenario = self.scenario_manager.create_scenario(
                scenario_id=f"sweep_{parameter_name}_{i}",
                name=f"{parameter_name}={value}",
                description=f"Parameter sweep: {parameter_name}={value}",
                parameters=params,
            )

            # Run scenario
            scenario_results = self.scenario_manager.run_scenario(
                scenario, self.simulator_factory
            )
            results[str(value)] = scenario_results

        logger.info(
            f"Parameter sweep completed for {parameter_name} "
            f"({len(values)} values)"
        )

        return results

    def multi_parameter_optimization(
        self,
        parameters: Dict[str, List[Any]],
        objective: str,
        maximize: bool = True,
    ) -> Scenario:
        """
        Test all combinations of parameters to find optimal configuration.

        Args:
            parameters: Dictionary of parameter_name -> list of values
            objective: Metric to optimize
            maximize: True to maximize, False to minimize

        Returns:
            Best scenario
        """
        # Generate all combinations
        param_names = list(parameters.keys())
        param_values = [parameters[name] for name in param_names]

        import itertools

        combinations = list(itertools.product(*param_values))

        logger.info(
            f"Testing {len(combinations)} parameter combinations "
            f"to optimize {objective}"
        )

        # Test each combination
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            scenario = self.scenario_manager.create_scenario(
                scenario_id=f"combo_{i}",
                name=f"Combo {i+1}",
                description=f"Parameters: {params}",
                parameters=params,
            )

            self.scenario_manager.run_scenario(scenario, self.simulator_factory)

        # Find best
        best = self.scenario_manager.find_best_scenario(objective, maximize)

        if best:
            logger.info(
                f"Optimal configuration found: {best.name}\n"
                f"Parameters: {best.parameters}\n"
                f"{objective}: {self.scenario_manager._get_nested_value(best.results, objective)}"
            )

        return best

    def sensitivity_analysis(
        self,
        parameters: Dict[str, List[float]],
        base_case: Dict[str, Any],
        metric: str,
    ) -> Dict[str, List[float]]:
        """
        Analyze sensitivity of metric to each parameter.

        Args:
            parameters: Parameters to test with variation ranges
            base_case: Base case parameter values
            metric: Metric to measure sensitivity

        Returns:
            Dictionary of parameter -> metric values
        """
        sensitivity = {}

        for param_name, values in parameters.items():
            metric_values = []

            for value in values:
                # Create scenario with one parameter varied
                params = base_case.copy()
                params[param_name] = value

                scenario = self.scenario_manager.create_scenario(
                    scenario_id=f"sens_{param_name}_{value}",
                    name=f"Sensitivity: {param_name}={value}",
                    description=f"Sensitivity analysis for {param_name}",
                    parameters=params,
                )

                results = self.scenario_manager.run_scenario(
                    scenario, self.simulator_factory
                )
                metric_value = self.scenario_manager._get_nested_value(results, metric)
                metric_values.append(metric_value)

            sensitivity[param_name] = metric_values

        logger.info(f"Sensitivity analysis completed for {len(parameters)} parameters")

        return sensitivity


def create_example_scenarios() -> ScenarioManager:
    """
    Create example scenarios for demonstration.

    Returns:
        ScenarioManager with example scenarios
    """
    manager = ScenarioManager()

    # Baseline scenario
    baseline = manager.create_scenario(
        scenario_id="baseline",
        name="Baseline",
        description="Normal production conditions",
        parameters={
            "product_arrival_rate": 60.0,  # One per minute
        },
    )
    manager.set_baseline(baseline)

    # High demand scenario
    manager.create_scenario(
        scenario_id="high_demand",
        name="High Demand",
        description="Increased product arrival rate",
        parameters={
            "product_arrival_rate": 30.0,  # Two per minute
        },
    )

    # Low defect rate scenario (better quality)
    manager.create_scenario(
        scenario_id="improved_quality",
        name="Improved Quality",
        description="Reduced defect rates at all stations",
        parameters={
            "station_Line_01_assembly_defect_rate": 0.01,  # Half of normal
            "station_Line_01_packaging_defect_rate": 0.01,
        },
    )

    # Fast processing scenario
    manager.create_scenario(
        scenario_id="fast_processing",
        name="Fast Processing",
        description="Reduced processing times",
        parameters={
            "station_Line_01_assembly_processing_time_mean": 20.0,  # Down from 30
            "station_Line_01_packaging_processing_time_mean": 10.0,  # Down from 15
        },
    )

    logger.info("Example scenarios created")

    return manager
