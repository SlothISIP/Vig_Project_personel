"""Factory Discrete-Event Simulator.

SimPy-based discrete-event simulation for factory digital twin.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import simpy
import uuid

from src.digital_twin.simulation.production_line import (
    ProductionLine,
    Product,
    WorkStation,
    create_sample_production_line,
)
from src.digital_twin.events.event_bus import get_event_bus
from src.digital_twin.events.event_types import Event, EventType
from src.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for factory simulation."""

    # Simulation parameters
    duration: float = 8.0 * 3600  # 8 hours in seconds
    product_arrival_rate: float = 60.0  # seconds between product arrivals
    product_types: List[str] = field(default_factory=lambda: ["ProductA", "ProductB"])

    # Random seed for reproducibility
    random_seed: Optional[int] = None

    # Real-time factor (1.0 = real-time, 0 = as fast as possible)
    realtime_factor: float = 0.0

    # Enable detailed logging
    verbose: bool = False


class FactorySimulator:
    """
    Discrete-event simulator for factory digital twin.

    Uses SimPy to simulate:
    - Product arrivals
    - Production line processing
    - Machine failures and repairs
    - Sensor data generation
    """

    def __init__(self, config: SimulationConfig, production_lines: List[ProductionLine]):
        """
        Initialize factory simulator.

        Args:
            config: Simulation configuration
            production_lines: List of production lines to simulate
        """
        self.config = config
        self.production_lines = {line.line_id: line for line in production_lines}

        # SimPy environment
        self.env = simpy.Environment()

        # Event bus for publishing events
        self.event_bus = get_event_bus()

        # Simulation state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.simulation_time = 0.0
        self.products_created = 0

        # Statistics
        self.total_products_introduced = 0
        self.total_products_completed = 0
        self.total_products_defective = 0

        # Callbacks
        self._step_callbacks: List[Callable[[float], None]] = []

        # Process initialization flag for step mode
        self._processes_initialized = False

        logger.info(
            f"Factory simulator initialized with {len(production_lines)} production lines"
        )

    def _product_generator(self, line: ProductionLine):
        """
        Generate products at specified arrival rate.

        Args:
            line: Production line to introduce products to
        """
        while True:
            # Wait for next arrival
            yield self.env.timeout(self.config.product_arrival_rate)

            # Create new product
            product_id = f"P{self.products_created:06d}"
            product_type = self.config.product_types[
                self.products_created % len(self.config.product_types)
            ]

            product = Product(product_id=product_id, product_type=product_type)

            # Introduce to production line
            line.introduce_product(product)
            self.products_created += 1
            self.total_products_introduced += 1

            if self.config.verbose:
                logger.info(
                    f"[t={self.env.now:.1f}s] Product {product_id} "
                    f"introduced to {line.line_id}"
                )

            # Publish event
            self._publish_event(
                EventType.PRODUCT_INTRODUCED,
                {
                    "product_id": product_id,
                    "product_type": product_type,
                    "line_id": line.line_id,
                },
            )

    def _station_processor(self, station: WorkStation, line: ProductionLine):
        """
        Process products at a workstation.

        Args:
            station: Workstation to process
            line: Production line the station belongs to
        """
        while True:
            # Try to start processing next product
            product = station.start_processing_next()

            if product is None:
                # No product to process, wait a bit
                yield self.env.timeout(1.0)
                continue

            # Simulate processing time
            yield self.env.timeout(station.processing_time_mean)

            # Complete processing
            station.complete_processing(product)

            if self.config.verbose:
                logger.info(
                    f"[t={self.env.now:.1f}s] Product {product.product_id} "
                    f"completed at {station.station_id}"
                )

            # Route to next station
            line.route_product(product)

            # Update statistics
            if product.status.value == "completed":
                self.total_products_completed += 1
            elif product.status.value == "defective":
                self.total_products_defective += 1

    def _sensor_monitor(self, station: WorkStation):
        """
        Periodically read sensors and publish data.

        Args:
            station: Workstation with sensors
        """
        while True:
            # Read all sensors
            readings = station.sensor_network.read_all()

            # Publish sensor readings
            for reading in readings:
                if reading.anomaly_score > 0.5:
                    self._publish_event(
                        EventType.ANOMALY_DETECTED,
                        {
                            "sensor_id": reading.sensor_id,
                            "sensor_type": reading.sensor_type.value,
                            "value": reading.value,
                            "anomaly_score": reading.anomaly_score,
                            "station_id": station.station_id,
                        },
                    )

            # Wait before next reading (e.g., every 10 seconds)
            yield self.env.timeout(10.0)

    def _failure_generator(self, station: WorkStation):
        """
        Simulate random machine failures.

        Args:
            station: Workstation to simulate failures for
        """
        while True:
            # Wait until next failure (MTBF)
            yield self.env.timeout(station.mtbf)

            # Trigger downtime
            station.trigger_downtime()

            if self.config.verbose:
                logger.warning(
                    f"[t={self.env.now:.1f}s] Station {station.station_id} failed!"
                )

            self._publish_event(
                EventType.MACHINE_FAILED,
                {"station_id": station.station_id, "mttr": station.mttr},
            )

            # Wait for repair (MTTR)
            yield self.env.timeout(station.mttr)

            # Perform maintenance
            station.perform_maintenance()

            if self.config.verbose:
                logger.info(
                    f"[t={self.env.now:.1f}s] Station {station.station_id} repaired"
                )

            self._publish_event(
                EventType.MAINTENANCE_COMPLETED, {"station_id": station.station_id}
            )

    def _simulation_monitor(self):
        """Monitor simulation progress and call callbacks."""
        while True:
            # Update simulation time
            self.simulation_time = self.env.now

            # Call step callbacks
            for callback in self._step_callbacks:
                try:
                    callback(self.simulation_time)
                except Exception as e:
                    logger.error(f"Error in simulation step callback: {e}")

            # Wait a bit
            yield self.env.timeout(60.0)  # Update every minute

    def _publish_event(self, event_type: EventType, data: Dict):
        """Publish simulation event."""
        event = Event(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            source="FactorySimulator",
            timestamp=datetime.now(),
            data=data,
        )
        self.event_bus.publish(event)

    def add_step_callback(self, callback: Callable[[float], None]) -> None:
        """
        Add callback to be called at each simulation step.

        Args:
            callback: Function that receives simulation time
        """
        self._step_callbacks.append(callback)

    def run(self) -> None:
        """Run the simulation."""
        if self.is_running:
            logger.warning("Simulation is already running")
            return

        self.is_running = True
        self.start_time = datetime.now()

        logger.info(
            f"Starting factory simulation for {self.config.duration}s "
            f"({self.config.duration/3600:.1f} hours)"
        )

        # Start processes for each production line
        for line in self.production_lines.values():
            # Product generator
            self.env.process(self._product_generator(line))

            # Station processors
            for station in line.stations.values():
                self.env.process(self._station_processor(station, line))
                self.env.process(self._sensor_monitor(station))
                self.env.process(self._failure_generator(station))

        # Simulation monitor
        self.env.process(self._simulation_monitor())

        # Run simulation
        try:
            self.env.run(until=self.config.duration)
            logger.info("Simulation completed successfully")
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self.is_running = False

        # Print final statistics
        self._print_statistics()

    def _initialize_processes(self) -> None:
        """
        Initialize SimPy processes for step-by-step simulation.

        This registers all generator processes with the SimPy environment
        so that step() can advance the simulation incrementally.
        """
        if self._processes_initialized:
            return

        # Start processes for each production line
        for line in self.production_lines.values():
            # Product generator
            self.env.process(self._product_generator(line))

            # Station processors
            for station in line.stations.values():
                self.env.process(self._station_processor(station, line))
                self.env.process(self._sensor_monitor(station))
                self.env.process(self._failure_generator(station))

        # Simulation monitor
        self.env.process(self._simulation_monitor())

        self._processes_initialized = True
        logger.debug("SimPy processes initialized for step mode")

    def step(self, duration: float = 1.0) -> None:
        """
        Run simulation for a single step.

        Args:
            duration: Step duration in seconds
        """
        if not self.is_running:
            self.is_running = True
            self.start_time = datetime.now()

        # Initialize processes on first step (CRITICAL FIX)
        if not self._processes_initialized:
            self._initialize_processes()

        self.env.run(until=self.env.now + duration)
        self.simulation_time = self.env.now

    def reset(self) -> None:
        """Reset simulation to initial state."""
        # Reset environment
        self.env = simpy.Environment()

        # Reset process initialization flag (CRITICAL for step mode)
        self._processes_initialized = False

        # Reset statistics
        self.simulation_time = 0.0
        self.products_created = 0
        self.total_products_introduced = 0
        self.total_products_completed = 0
        self.total_products_defective = 0

        # Reset production lines
        for line in self.production_lines.values():
            line.products.clear()
            line.completed_products.clear()
            line.defective_products.clear()

            for station in line.stations.values():
                station.machine_state.health_score = 1.0
                station.machine_state.cycle_count = 0
                station.machine_state.defect_count = 0
                station.sensor_network.reset_all()

        self.is_running = False
        logger.info("Simulation reset")

    def get_statistics(self) -> Dict:
        """Get comprehensive simulation statistics."""
        line_stats = {
            line_id: line.get_statistics()
            for line_id, line in self.production_lines.items()
        }

        return {
            "simulation_time": self.simulation_time,
            "duration": self.config.duration,
            "progress": self.simulation_time / self.config.duration,
            "total_products_introduced": self.total_products_introduced,
            "total_products_completed": self.total_products_completed,
            "total_products_defective": self.total_products_defective,
            "overall_yield": (
                self.total_products_completed
                / (self.total_products_completed + self.total_products_defective)
                if (self.total_products_completed + self.total_products_defective) > 0
                else 0.0
            ),
            "production_lines": line_stats,
        }

    def _print_statistics(self) -> None:
        """Print simulation statistics."""
        stats = self.get_statistics()

        logger.info("=" * 60)
        logger.info("SIMULATION STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Simulation Time: {stats['simulation_time']:.1f}s")
        logger.info(f"Products Introduced: {stats['total_products_introduced']}")
        logger.info(f"Products Completed: {stats['total_products_completed']}")
        logger.info(f"Products Defective: {stats['total_products_defective']}")
        logger.info(f"Overall Yield: {stats['overall_yield']:.2%}")
        logger.info("=" * 60)


def create_factory_simulator(
    num_lines: int = 1, simulation_hours: float = 8.0
) -> FactorySimulator:
    """
    Create a factory simulator with sample production lines.

    Args:
        num_lines: Number of production lines
        simulation_hours: Simulation duration in hours

    Returns:
        Configured factory simulator
    """
    # Create production lines
    lines = [create_sample_production_line(f"Line_{i+1:02d}") for i in range(num_lines)]

    # Create configuration
    config = SimulationConfig(
        duration=simulation_hours * 3600,
        product_arrival_rate=60.0,  # One product per minute
        verbose=False,
    )

    simulator = FactorySimulator(config, lines)
    logger.info(
        f"Factory simulator created with {num_lines} lines, "
        f"{simulation_hours}h simulation"
    )

    return simulator
