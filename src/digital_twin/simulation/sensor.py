"""IoT Sensor Simulation for Digital Twin.

Simulates various types of industrial sensors with realistic noise and patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable
import random
import math

from src.core.logging import get_logger

logger = get_logger(__name__)


class SensorType(Enum):
    """Types of IoT sensors."""

    TEMPERATURE = "temperature"
    VIBRATION = "vibration"
    PRESSURE = "pressure"
    CURRENT = "current"
    VOLTAGE = "voltage"
    SPEED = "speed"
    POSITION = "position"
    DEFECT_RATE = "defect_rate"


@dataclass
class SensorReading:
    """A single sensor reading."""

    sensor_id: str
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: datetime
    quality: float = 1.0  # 0-1, data quality indicator
    anomaly_score: float = 0.0  # 0-1, anomaly detection score


@dataclass
class IoTSensor:
    """
    Simulated IoT sensor with realistic behavior.

    Generates sensor readings with:
    - Base value with configurable range
    - Random noise
    - Periodic patterns (optional)
    - Drift over time (optional)
    - Anomaly injection (optional)
    """

    sensor_id: str
    sensor_type: SensorType
    unit: str
    base_value: float
    noise_std: float = 0.1
    drift_rate: float = 0.0  # Value change per hour
    periodic_amplitude: float = 0.0  # Amplitude of periodic variation
    periodic_frequency: float = 1.0  # Cycles per hour
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    anomaly_probability: float = 0.01  # Probability of anomaly per reading

    # State
    current_value: float = field(init=False)
    last_update: Optional[datetime] = None
    total_drift: float = 0.0
    readings_count: int = 0

    def __post_init__(self):
        """Initialize sensor state."""
        self.current_value = self.base_value

    def read(self) -> SensorReading:
        """
        Generate a sensor reading.

        Returns:
            SensorReading with current sensor data
        """
        now = datetime.now()

        # Calculate time since last reading (in hours)
        if self.last_update:
            hours_elapsed = (now - self.last_update).total_seconds() / 3600
        else:
            hours_elapsed = 0

        # Apply drift
        self.total_drift += self.drift_rate * hours_elapsed

        # Calculate base value with drift
        value = self.base_value + self.total_drift

        # Add periodic component
        if self.periodic_amplitude > 0:
            time_in_hours = self.readings_count / 60  # Assuming 1 reading per minute
            periodic = self.periodic_amplitude * math.sin(
                2 * math.pi * self.periodic_frequency * time_in_hours
            )
            value += periodic

        # Add noise
        noise = random.gauss(0, self.noise_std)
        value += noise

        # Check for anomaly
        anomaly_score = 0.0
        if random.random() < self.anomaly_probability:
            # Inject anomaly (spike or drop)
            anomaly_magnitude = random.uniform(2, 5) * self.noise_std
            if random.random() < 0.5:
                value += anomaly_magnitude
            else:
                value -= anomaly_magnitude
            anomaly_score = random.uniform(0.7, 1.0)
            logger.warning(f"Anomaly injected in sensor {self.sensor_id}")

        # Clamp to min/max
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)

        # Data quality (degraded if anomaly or out of expected range)
        quality = 1.0 - anomaly_score

        self.current_value = value
        self.last_update = now
        self.readings_count += 1

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            value=value,
            unit=self.unit,
            timestamp=now,
            quality=quality,
            anomaly_score=anomaly_score,
        )

    def reset(self) -> None:
        """Reset sensor to initial state."""
        self.current_value = self.base_value
        self.total_drift = 0.0
        self.readings_count = 0
        self.last_update = None
        logger.info(f"Sensor {self.sensor_id} reset")


class SensorNetwork:
    """
    Network of IoT sensors for a production line.

    Manages multiple sensors and provides aggregated readings.
    """

    def __init__(self, network_id: str):
        """
        Initialize sensor network.

        Args:
            network_id: Unique identifier for the network
        """
        self.network_id = network_id
        self.sensors: Dict[str, IoTSensor] = {}
        self._reading_callbacks: List[Callable[[SensorReading], None]] = []

    def add_sensor(self, sensor: IoTSensor) -> None:
        """
        Add a sensor to the network.

        Args:
            sensor: IoT sensor to add
        """
        self.sensors[sensor.sensor_id] = sensor
        logger.info(
            f"Sensor {sensor.sensor_id} ({sensor.sensor_type.value}) "
            f"added to network {self.network_id}"
        )

    def remove_sensor(self, sensor_id: str) -> None:
        """
        Remove a sensor from the network.

        Args:
            sensor_id: ID of sensor to remove
        """
        if sensor_id in self.sensors:
            del self.sensors[sensor_id]
            logger.info(f"Sensor {sensor_id} removed from network {self.network_id}")

    def read_all(self) -> List[SensorReading]:
        """
        Read all sensors in the network.

        Returns:
            List of sensor readings
        """
        readings = []
        for sensor in self.sensors.values():
            reading = sensor.read()
            readings.append(reading)

            # Call callbacks
            for callback in self._reading_callbacks:
                try:
                    callback(reading)
                except Exception as e:
                    logger.error(f"Error in sensor reading callback: {e}")

        return readings

    def read_by_type(self, sensor_type: SensorType) -> List[SensorReading]:
        """
        Read all sensors of a specific type.

        Args:
            sensor_type: Type of sensors to read

        Returns:
            List of sensor readings
        """
        readings = []
        for sensor in self.sensors.values():
            if sensor.sensor_type == sensor_type:
                readings.append(sensor.read())
        return readings

    def get_sensor(self, sensor_id: str) -> Optional[IoTSensor]:
        """
        Get a sensor by ID.

        Args:
            sensor_id: Sensor ID

        Returns:
            IoT sensor or None if not found
        """
        return self.sensors.get(sensor_id)

    def subscribe(self, callback: Callable[[SensorReading], None]) -> None:
        """
        Subscribe to sensor readings.

        Args:
            callback: Function to call when sensor is read
        """
        self._reading_callbacks.append(callback)

    def reset_all(self) -> None:
        """Reset all sensors to initial state."""
        for sensor in self.sensors.values():
            sensor.reset()
        logger.info(f"All sensors in network {self.network_id} reset")

    def get_statistics(self) -> Dict[str, any]:
        """
        Get network statistics.

        Returns:
            Dictionary with network statistics
        """
        total_sensors = len(self.sensors)
        by_type = {}
        for sensor in self.sensors.values():
            sensor_type = sensor.sensor_type.value
            by_type[sensor_type] = by_type.get(sensor_type, 0) + 1

        return {
            "network_id": self.network_id,
            "total_sensors": total_sensors,
            "sensors_by_type": by_type,
            "total_readings": sum(s.readings_count for s in self.sensors.values()),
        }


def create_standard_sensor_network(
    machine_id: str, include_defect_sensor: bool = True
) -> SensorNetwork:
    """
    Create a standard sensor network for a production machine.

    Args:
        machine_id: Machine identifier
        include_defect_sensor: Whether to include defect rate sensor

    Returns:
        Configured sensor network
    """
    network = SensorNetwork(network_id=f"{machine_id}_network")

    # Temperature sensor (°C)
    temp_sensor = IoTSensor(
        sensor_id=f"{machine_id}_temp",
        sensor_type=SensorType.TEMPERATURE,
        unit="°C",
        base_value=65.0,
        noise_std=2.0,
        drift_rate=0.5,  # Slowly heats up
        periodic_amplitude=3.0,
        periodic_frequency=2.0,  # 2 cycles per hour
        min_value=20.0,
        max_value=100.0,
        anomaly_probability=0.02,
    )
    network.add_sensor(temp_sensor)

    # Vibration sensor (mm/s)
    vib_sensor = IoTSensor(
        sensor_id=f"{machine_id}_vib",
        sensor_type=SensorType.VIBRATION,
        unit="mm/s",
        base_value=2.5,
        noise_std=0.3,
        periodic_amplitude=0.5,
        periodic_frequency=10.0,  # High frequency
        min_value=0.0,
        max_value=20.0,
        anomaly_probability=0.015,
    )
    network.add_sensor(vib_sensor)

    # Current sensor (A)
    current_sensor = IoTSensor(
        sensor_id=f"{machine_id}_current",
        sensor_type=SensorType.CURRENT,
        unit="A",
        base_value=15.0,
        noise_std=0.5,
        periodic_amplitude=2.0,
        periodic_frequency=1.0,
        min_value=0.0,
        max_value=50.0,
        anomaly_probability=0.01,
    )
    network.add_sensor(current_sensor)

    # Speed sensor (RPM)
    speed_sensor = IoTSensor(
        sensor_id=f"{machine_id}_speed",
        sensor_type=SensorType.SPEED,
        unit="RPM",
        base_value=1200.0,
        noise_std=10.0,
        min_value=0.0,
        max_value=2000.0,
        anomaly_probability=0.01,
    )
    network.add_sensor(speed_sensor)

    # Defect rate sensor (%)
    if include_defect_sensor:
        defect_sensor = IoTSensor(
            sensor_id=f"{machine_id}_defect_rate",
            sensor_type=SensorType.DEFECT_RATE,
            unit="%",
            base_value=2.0,
            noise_std=0.5,
            drift_rate=0.1,  # Slowly increases if not maintained
            min_value=0.0,
            max_value=100.0,
            anomaly_probability=0.02,
        )
        network.add_sensor(defect_sensor)

    logger.info(f"Standard sensor network created for {machine_id}")
    return network
