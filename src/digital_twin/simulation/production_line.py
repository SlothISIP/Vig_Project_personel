"""Production Line Simulation Models.

Models for simulating production lines with workstations, buffers, and routing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set
from queue import Queue
import random

from src.digital_twin.state.machine_state import MachineState, MachineStatus
from src.digital_twin.simulation.sensor import SensorNetwork, create_standard_sensor_network
from src.core.logging import get_logger

logger = get_logger(__name__)


class ProductStatus(Enum):
    """Status of a product in the production line."""

    WAITING = "waiting"
    IN_PROCESS = "in_process"
    COMPLETED = "completed"
    DEFECTIVE = "defective"
    SCRAPPED = "scrapped"


@dataclass
class Product:
    """
    A product being manufactured.

    Tracks the product's journey through the production line.
    """

    product_id: str
    product_type: str
    status: ProductStatus = ProductStatus.WAITING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_station: Optional[str] = None
    visited_stations: List[str] = field(default_factory=list)
    processing_times: Dict[str, float] = field(default_factory=dict)
    is_defective: bool = False
    defect_detected_at: Optional[str] = None

    def start_processing(self, station_id: str) -> None:
        """Mark product as started at a station."""
        self.status = ProductStatus.IN_PROCESS
        self.current_station = station_id
        if self.started_at is None:
            self.started_at = datetime.now()
        self.visited_stations.append(station_id)

    def complete_processing(self, processing_time: float) -> None:
        """Mark product as completed at current station."""
        if self.current_station:
            self.processing_times[self.current_station] = processing_time
        self.status = ProductStatus.WAITING

    def mark_completed(self) -> None:
        """Mark product as completed."""
        self.status = ProductStatus.COMPLETED
        self.completed_at = datetime.now()
        self.current_station = None

    def mark_defective(self, station_id: str) -> None:
        """Mark product as defective."""
        self.is_defective = True
        self.status = ProductStatus.DEFECTIVE
        self.defect_detected_at = station_id

    def get_total_processing_time(self) -> float:
        """Get total processing time across all stations."""
        return sum(self.processing_times.values())

    def get_throughput_time(self) -> Optional[float]:
        """Get total time from start to completion (seconds)."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


@dataclass
class WorkStation:
    """
    A workstation in the production line.

    Processes products with configurable:
    - Processing time (mean + variance)
    - Defect rate
    - Capacity (parallel processing)
    - Downtime probability
    """

    station_id: str
    station_type: str
    processing_time_mean: float  # seconds
    processing_time_std: float = 5.0  # seconds
    defect_rate: float = 0.02  # 2% base defect rate
    capacity: int = 1  # Number of products that can be processed in parallel
    mtbf: float = 10000.0  # Mean Time Between Failures (seconds)
    mttr: float = 600.0  # Mean Time To Repair (seconds)

    # State
    machine_state: MachineState = field(init=False)
    sensor_network: SensorNetwork = field(init=False)
    buffer: Queue = field(default_factory=Queue)
    products_in_process: Set[str] = field(default_factory=set)
    total_processed: int = 0
    total_defects: int = 0
    downtime_events: int = 0

    def __post_init__(self):
        """Initialize workstation state."""
        self.machine_state = MachineState(
            machine_id=self.station_id, machine_type=self.station_type
        )
        self.sensor_network = create_standard_sensor_network(
            self.station_id, include_defect_sensor=True
        )

    def can_accept_product(self) -> bool:
        """Check if station can accept a new product."""
        return len(self.products_in_process) < self.capacity

    def add_to_buffer(self, product: Product) -> None:
        """Add product to station buffer."""
        self.buffer.put(product)
        logger.debug(f"Product {product.product_id} added to {self.station_id} buffer")

    def start_processing_next(self) -> Optional[Product]:
        """
        Start processing the next product from buffer.

        Returns:
            Product being processed, or None if buffer is empty
        """
        if not self.buffer.empty() and self.can_accept_product():
            product = self.buffer.get()
            product.start_processing(self.station_id)
            self.products_in_process.add(product.product_id)
            self.machine_state.update_status(MachineStatus.RUNNING)
            self.machine_state.increment_cycle()
            logger.info(
                f"Station {self.station_id} started processing {product.product_id}"
            )
            return product
        return None

    def complete_processing(self, product: Product) -> None:
        """
        Complete processing of a product.

        Args:
            product: Product that finished processing
        """
        # Calculate actual processing time
        processing_time = max(
            1.0, random.gauss(self.processing_time_mean, self.processing_time_std)
        )

        # Check for defect
        # Defect rate increases with machine health degradation
        effective_defect_rate = self.defect_rate * (2.0 - self.machine_state.health_score)

        is_defect = random.random() < effective_defect_rate

        if is_defect:
            product.mark_defective(self.station_id)
            self.total_defects += 1
            self.machine_state.report_defect()
            logger.warning(
                f"Defect detected in {product.product_id} at {self.station_id}"
            )
        else:
            product.complete_processing(processing_time)

        self.products_in_process.discard(product.product_id)
        self.total_processed += 1

        # Update status
        if len(self.products_in_process) == 0:
            self.machine_state.update_status(MachineStatus.IDLE)

    def trigger_downtime(self) -> None:
        """Trigger a downtime event."""
        self.machine_state.update_status(MachineStatus.ERROR)
        self.downtime_events += 1
        logger.error(f"Station {self.station_id} is down!")

    def perform_maintenance(self) -> None:
        """Perform maintenance on the station."""
        self.machine_state.perform_maintenance()
        logger.info(f"Maintenance performed on {self.station_id}")

    def get_statistics(self) -> Dict[str, any]:
        """Get station statistics."""
        return {
            "station_id": self.station_id,
            "station_type": self.station_type,
            "status": self.machine_state.status.value,
            "health_score": self.machine_state.health_score,
            "total_processed": self.total_processed,
            "total_defects": self.total_defects,
            "defect_rate": (
                self.total_defects / self.total_processed
                if self.total_processed > 0
                else 0.0
            ),
            "buffer_size": self.buffer.qsize(),
            "in_process": len(self.products_in_process),
            "downtime_events": self.downtime_events,
            "utilization": len(self.products_in_process) / self.capacity,
        }


class ProductionLine:
    """
    A production line consisting of multiple workstations.

    Models the flow of products through sequential processing stations.
    """

    def __init__(self, line_id: str, stations: List[WorkStation]):
        """
        Initialize production line.

        Args:
            line_id: Unique identifier
            stations: List of workstations in order
        """
        self.line_id = line_id
        self.stations = {station.station_id: station for station in stations}
        self.station_order = [station.station_id for station in stations]
        self.products: Dict[str, Product] = {}
        self.completed_products: List[Product] = []
        self.defective_products: List[Product] = []

    def introduce_product(self, product: Product) -> None:
        """
        Introduce a new product to the production line.

        Args:
            product: Product to introduce
        """
        self.products[product.product_id] = product

        # Add to first station buffer
        if self.station_order:
            first_station = self.stations[self.station_order[0]]
            first_station.add_to_buffer(product)
            logger.info(f"Product {product.product_id} introduced to {self.line_id}")

    def route_product(self, product: Product) -> None:
        """
        Route product to next station or complete it.

        Args:
            product: Product to route
        """
        if product.current_station is None:
            return

        current_idx = self.station_order.index(product.current_station)

        # Check if defective
        if product.is_defective:
            self.defective_products.append(product)
            del self.products[product.product_id]
            logger.warning(
                f"Product {product.product_id} marked as defective and removed"
            )
            return

        # Check if last station
        if current_idx == len(self.station_order) - 1:
            # Product completed
            product.mark_completed()
            self.completed_products.append(product)
            del self.products[product.product_id]
            logger.info(f"Product {product.product_id} completed production")
        else:
            # Route to next station
            next_station_id = self.station_order[current_idx + 1]
            next_station = self.stations[next_station_id]
            next_station.add_to_buffer(product)
            product.current_station = None

    def get_station(self, station_id: str) -> Optional[WorkStation]:
        """Get a workstation by ID."""
        return self.stations.get(station_id)

    def get_statistics(self) -> Dict[str, any]:
        """Get production line statistics."""
        total_in_process = len(self.products)
        total_completed = len(self.completed_products)
        total_defective = len(self.defective_products)

        # Calculate throughput times
        throughput_times = [
            p.get_throughput_time()
            for p in self.completed_products
            if p.get_throughput_time() is not None
        ]
        avg_throughput = (
            sum(throughput_times) / len(throughput_times) if throughput_times else 0.0
        )

        # Station statistics
        station_stats = {
            station_id: station.get_statistics()
            for station_id, station in self.stations.items()
        }

        return {
            "line_id": self.line_id,
            "total_in_process": total_in_process,
            "total_completed": total_completed,
            "total_defective": total_defective,
            "overall_yield": (
                total_completed / (total_completed + total_defective)
                if (total_completed + total_defective) > 0
                else 0.0
            ),
            "average_throughput_time": avg_throughput,
            "stations": station_stats,
        }

    def perform_maintenance_all(self) -> None:
        """Perform maintenance on all stations."""
        for station in self.stations.values():
            station.perform_maintenance()
        logger.info(f"Maintenance performed on all stations in {self.line_id}")


def create_sample_production_line(line_id: str = "Line_01") -> ProductionLine:
    """
    Create a sample production line with typical manufacturing stations.

    Args:
        line_id: Production line identifier

    Returns:
        Configured production line
    """
    # Station 1: Raw material loading
    loading_station = WorkStation(
        station_id=f"{line_id}_loading",
        station_type="loading",
        processing_time_mean=10.0,
        processing_time_std=2.0,
        defect_rate=0.01,
        capacity=2,
    )

    # Station 2: Assembly
    assembly_station = WorkStation(
        station_id=f"{line_id}_assembly",
        station_type="assembly",
        processing_time_mean=30.0,
        processing_time_std=5.0,
        defect_rate=0.03,
        capacity=1,
    )

    # Station 3: Quality inspection (Vision AI)
    inspection_station = WorkStation(
        station_id=f"{line_id}_inspection",
        station_type="inspection",
        processing_time_mean=5.0,
        processing_time_std=1.0,
        defect_rate=0.005,  # Very low defect rate for inspection itself
        capacity=1,
    )

    # Station 4: Packaging
    packaging_station = WorkStation(
        station_id=f"{line_id}_packaging",
        station_type="packaging",
        processing_time_mean=15.0,
        processing_time_std=3.0,
        defect_rate=0.02,
        capacity=2,
    )

    production_line = ProductionLine(
        line_id=line_id,
        stations=[loading_station, assembly_station, inspection_station, packaging_station],
    )

    logger.info(f"Sample production line {line_id} created with 4 stations")
    return production_line
