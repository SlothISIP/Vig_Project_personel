"""Tests for digital twin simulation components."""

import pytest
from datetime import datetime

from src.digital_twin.simulation.sensor import (
    IoTSensor,
    SensorType,
    SensorNetwork,
    create_standard_sensor_network,
)
from src.digital_twin.simulation.production_line import (
    Product,
    ProductStatus,
    WorkStation,
    ProductionLine,
    create_sample_production_line,
)
from src.digital_twin.simulation.simulator import (
    FactorySimulator,
    SimulationConfig,
    create_factory_simulator,
)
from src.digital_twin.simulation.scenario import (
    Scenario,
    ScenarioManager,
    WhatIfAnalysis,
)


class TestIoTSensor:
    """Tests for IoT sensors."""

    def test_create_sensor(self):
        """Test sensor creation."""
        sensor = IoTSensor(
            sensor_id="temp_01",
            sensor_type=SensorType.TEMPERATURE,
            unit="°C",
            base_value=65.0,
            noise_std=2.0,
        )

        assert sensor.sensor_id == "temp_01"
        assert sensor.sensor_type == SensorType.TEMPERATURE
        assert sensor.base_value == 65.0

    def test_read_sensor(self):
        """Test sensor reading."""
        sensor = IoTSensor(
            sensor_id="temp_01",
            sensor_type=SensorType.TEMPERATURE,
            unit="°C",
            base_value=65.0,
            noise_std=2.0,
            min_value=20.0,
            max_value=100.0,
        )

        reading = sensor.read()

        assert reading.sensor_id == "temp_01"
        assert reading.sensor_type == SensorType.TEMPERATURE
        assert reading.unit == "°C"
        assert 20.0 <= reading.value <= 100.0
        assert reading.quality > 0

    def test_sensor_drift(self):
        """Test sensor drift over time."""
        sensor = IoTSensor(
            sensor_id="temp_01",
            sensor_type=SensorType.TEMPERATURE,
            unit="°C",
            base_value=65.0,
            drift_rate=1.0,  # 1 degree per hour
            noise_std=0.1,  # Low noise
        )

        # First reading
        reading1 = sensor.read()
        initial_value = reading1.value

        # Simulate 1 hour passed (set last_update)
        import time
        from datetime import timedelta

        sensor.last_update = datetime.now() - timedelta(hours=1)

        # Second reading
        reading2 = sensor.read()

        # Should have drifted up
        assert reading2.value > initial_value

    def test_sensor_reset(self):
        """Test sensor reset."""
        sensor = IoTSensor(
            sensor_id="temp_01",
            sensor_type=SensorType.TEMPERATURE,
            unit="°C",
            base_value=65.0,
        )

        # Take some readings
        for _ in range(5):
            sensor.read()

        assert sensor.readings_count == 5

        # Reset
        sensor.reset()

        assert sensor.readings_count == 0
        assert sensor.total_drift == 0.0


class TestSensorNetwork:
    """Tests for sensor network."""

    def test_create_network(self):
        """Test network creation."""
        network = SensorNetwork(network_id="Network_01")

        assert network.network_id == "Network_01"
        assert len(network.sensors) == 0

    def test_add_remove_sensor(self):
        """Test adding and removing sensors."""
        network = SensorNetwork(network_id="Network_01")

        sensor = IoTSensor(
            sensor_id="temp_01",
            sensor_type=SensorType.TEMPERATURE,
            unit="°C",
            base_value=65.0,
        )

        network.add_sensor(sensor)
        assert len(network.sensors) == 1

        network.remove_sensor("temp_01")
        assert len(network.sensors) == 0

    def test_read_all(self):
        """Test reading all sensors."""
        network = create_standard_sensor_network("Machine_01")

        readings = network.read_all()

        # Standard network has 5 sensors
        assert len(readings) == 5

    def test_get_statistics(self):
        """Test network statistics."""
        network = create_standard_sensor_network("Machine_01")

        stats = network.get_statistics()

        assert stats["network_id"] == "Machine_01_network"
        assert stats["total_sensors"] == 5


class TestProduct:
    """Tests for Product."""

    def test_create_product(self):
        """Test product creation."""
        product = Product(product_id="P001", product_type="ProductA")

        assert product.product_id == "P001"
        assert product.status == ProductStatus.WAITING
        assert not product.is_defective

    def test_product_processing(self):
        """Test product processing flow."""
        product = Product(product_id="P001", product_type="ProductA")

        # Start processing
        product.start_processing("Station_01")
        assert product.status == ProductStatus.IN_PROCESS
        assert product.current_station == "Station_01"

        # Complete processing
        product.complete_processing(processing_time=10.0)
        assert product.processing_times["Station_01"] == 10.0

    def test_mark_defective(self):
        """Test marking product as defective."""
        product = Product(product_id="P001", product_type="ProductA")

        product.mark_defective("Station_02")

        assert product.is_defective
        assert product.status == ProductStatus.DEFECTIVE
        assert product.defect_detected_at == "Station_02"


class TestWorkStation:
    """Tests for WorkStation."""

    def test_create_station(self):
        """Test station creation."""
        station = WorkStation(
            station_id="S01",
            station_type="assembly",
            processing_time_mean=30.0,
            capacity=2,
        )

        assert station.station_id == "S01"
        assert station.capacity == 2
        assert station.total_processed == 0

    def test_buffer_operations(self):
        """Test buffer operations."""
        station = WorkStation(
            station_id="S01",
            station_type="assembly",
            processing_time_mean=30.0,
            capacity=1,
        )

        product = Product(product_id="P001", product_type="ProductA")

        # Add to buffer
        station.add_to_buffer(product)
        assert station.buffer.qsize() == 1

        # Start processing
        processed = station.start_processing_next()
        assert processed is not None
        assert processed.product_id == "P001"
        assert station.buffer.qsize() == 0


class TestProductionLine:
    """Tests for ProductionLine."""

    def test_create_line(self):
        """Test production line creation."""
        line = create_sample_production_line("Line_01")

        assert line.line_id == "Line_01"
        assert len(line.stations) == 4  # Standard line has 4 stations

    def test_introduce_product(self):
        """Test introducing product to line."""
        line = create_sample_production_line("Line_01")

        product = Product(product_id="P001", product_type="ProductA")
        line.introduce_product(product)

        assert "P001" in line.products
        # Product should be in first station's buffer
        first_station = list(line.stations.values())[0]
        assert first_station.buffer.qsize() == 1


class TestScenario:
    """Tests for Scenario."""

    def test_create_scenario(self):
        """Test scenario creation."""
        scenario = Scenario(
            scenario_id="test_01",
            name="Test Scenario",
            description="Test description",
            parameters={"param1": 10},
        )

        assert scenario.scenario_id == "test_01"
        assert scenario.parameters["param1"] == 10

    def test_scenario_to_dict(self):
        """Test scenario serialization."""
        scenario = Scenario(
            scenario_id="test_01",
            name="Test",
            description="Test",
            parameters={"param1": 10},
        )

        data = scenario.to_dict()

        assert data["scenario_id"] == "test_01"
        assert data["parameters"]["param1"] == 10


class TestScenarioManager:
    """Tests for ScenarioManager."""

    def test_create_scenario_manager(self):
        """Test scenario manager creation."""
        manager = ScenarioManager()

        assert len(manager.scenarios) == 0

    def test_add_scenario(self):
        """Test adding scenario."""
        manager = ScenarioManager()

        scenario = manager.create_scenario(
            scenario_id="s1",
            name="Scenario 1",
            description="Test",
            parameters={"param1": 10},
        )

        assert "s1" in manager.scenarios
        assert manager.scenarios["s1"] == scenario

    def test_compare_scenarios(self):
        """Test scenario comparison."""
        manager = ScenarioManager()

        # Create scenarios with mock results
        s1 = manager.create_scenario("s1", "S1", "Test", {})
        s1.results = {"overall_yield": 0.95, "total_products_completed": 100}

        s2 = manager.create_scenario("s2", "S2", "Test", {})
        s2.results = {"overall_yield": 0.90, "total_products_completed": 95}

        comparison = manager.compare_scenarios(
            metrics=["overall_yield", "total_products_completed"]
        )

        assert "s1" in comparison
        assert comparison["s1"]["overall_yield"] == 0.95

    def test_find_best_scenario(self):
        """Test finding best scenario."""
        manager = ScenarioManager()

        s1 = manager.create_scenario("s1", "S1", "Test", {})
        s1.results = {"overall_yield": 0.95}

        s2 = manager.create_scenario("s2", "S2", "Test", {})
        s2.results = {"overall_yield": 0.90}

        best = manager.find_best_scenario("overall_yield", maximize=True)

        assert best.scenario_id == "s1"


class TestSimulation:
    """Integration tests for simulation."""

    def test_short_simulation(self):
        """Test running a short simulation."""
        # Create small simulation (10 seconds)
        config = SimulationConfig(
            duration=10.0,  # 10 seconds
            product_arrival_rate=5.0,  # One every 5 seconds
            verbose=False,
        )

        line = create_sample_production_line("Test_Line")
        simulator = FactorySimulator(config, [line])

        # Run simulation
        simulator.run()

        # Check statistics
        stats = simulator.get_statistics()

        assert stats["simulation_time"] == 10.0
        assert stats["total_products_introduced"] >= 0  # At least tried to introduce

    def test_simulator_reset(self):
        """Test simulator reset."""
        simulator = create_factory_simulator(num_lines=1, simulation_hours=0.01)

        # Run briefly
        simulator.step(1.0)

        # Reset
        simulator.reset()

        assert simulator.simulation_time == 0.0
        assert simulator.products_created == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
