"""Tests for digital twin machine state."""

import pytest
from datetime import datetime, timedelta

from src.digital_twin.state.machine_state import (
    MachineState,
    MachineStatus,
    FactoryState,
)


class TestMachineState:
    """Tests for MachineState."""

    def test_create_machine(self):
        """Test machine creation."""
        machine = MachineState(
            machine_id="M001",
            machine_type="assembly",
        )

        assert machine.machine_id == "M001"
        assert machine.machine_type == "assembly"
        assert machine.status == MachineStatus.IDLE
        assert machine.health_score == 1.0
        assert machine.cycle_count == 0
        assert machine.defect_count == 0

    def test_update_status(self):
        """Test status update."""
        machine = MachineState(machine_id="M001", machine_type="assembly")

        old_time = machine.updated_at
        machine.update_status(MachineStatus.RUNNING)

        assert machine.status == MachineStatus.RUNNING
        assert machine.updated_at > old_time

    def test_increment_cycle(self):
        """Test cycle increment."""
        machine = MachineState(machine_id="M001", machine_type="assembly")

        machine.increment_cycle()
        assert machine.cycle_count == 1

        machine.increment_cycle()
        assert machine.cycle_count == 2

    def test_report_defect(self):
        """Test defect reporting."""
        machine = MachineState(machine_id="M001", machine_type="assembly")

        machine.increment_cycle()
        machine.increment_cycle()
        machine.report_defect()

        assert machine.defect_count == 1
        assert machine.last_defect_time is not None
        assert machine.health_score == 0.5  # 1 defect in 2 cycles

    def test_health_score_degredation(self):
        """Test health score degrades with defects."""
        machine = MachineState(machine_id="M001", machine_type="assembly")

        # Run cycles without defects
        for _ in range(10):
            machine.increment_cycle()

        assert machine.health_score == 1.0

        # Report defects
        for _ in range(5):
            machine.report_defect()

        # Health should degrade (5 defects in 10 cycles = 0.5 health)
        assert machine.health_score == 0.5
        assert machine.status == MachineStatus.WARNING  # Triggers warning

    def test_maintenance(self):
        """Test maintenance."""
        machine = MachineState(machine_id="M001", machine_type="assembly")

        # Degrade health
        for _ in range(10):
            machine.increment_cycle()
            if _ < 5:
                machine.report_defect()

        assert machine.health_score < 1.0

        # Perform maintenance
        machine.perform_maintenance()

        assert machine.health_score == 1.0
        assert machine.defect_count == 0
        assert machine.status == MachineStatus.IDLE
        assert machine.last_maintenance is not None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        machine = MachineState(machine_id="M001", machine_type="assembly")

        data = machine.to_dict()

        assert data["machine_id"] == "M001"
        assert data["machine_type"] == "assembly"
        assert data["status"] == "idle"
        assert data["health_score"] == 1.0
        assert "defect_rate" in data


class TestFactoryState:
    """Tests for FactoryState."""

    def test_create_factory(self):
        """Test factory creation."""
        factory = FactoryState(factory_id="Factory_01")

        assert factory.factory_id == "Factory_01"
        assert len(factory.machines) == 0

    def test_add_machine(self):
        """Test adding machines."""
        factory = FactoryState(factory_id="Factory_01")

        machine1 = MachineState(machine_id="M001", machine_type="assembly")
        machine2 = MachineState(machine_id="M002", machine_type="inspection")

        factory.add_machine(machine1)
        factory.add_machine(machine2)

        assert len(factory.machines) == 2
        assert factory.get_machine("M001") == machine1
        assert factory.get_machine("M002") == machine2

    def test_overall_health(self):
        """Test overall health calculation."""
        factory = FactoryState(factory_id="Factory_01")

        m1 = MachineState(machine_id="M001", machine_type="assembly")
        m2 = MachineState(machine_id="M002", machine_type="inspection")

        m1.health_score = 1.0
        m2.health_score = 0.5

        factory.add_machine(m1)
        factory.add_machine(m2)

        assert factory.get_overall_health() == 0.75  # Average of 1.0 and 0.5

    def test_get_active_machines(self):
        """Test getting active machines."""
        factory = FactoryState(factory_id="Factory_01")

        m1 = MachineState(machine_id="M001", machine_type="assembly")
        m2 = MachineState(machine_id="M002", machine_type="inspection")
        m3 = MachineState(machine_id="M003", machine_type="packaging")

        m1.update_status(MachineStatus.RUNNING)
        m2.update_status(MachineStatus.IDLE)
        m3.update_status(MachineStatus.RUNNING)

        factory.add_machine(m1)
        factory.add_machine(m2)
        factory.add_machine(m3)

        active = factory.get_active_machines()
        assert len(active) == 2
        assert all(m.status == MachineStatus.RUNNING for m in active)

    def test_get_statistics(self):
        """Test statistics generation."""
        factory = FactoryState(factory_id="Factory_01")

        m1 = MachineState(machine_id="M001", machine_type="assembly")
        m2 = MachineState(machine_id="M002", machine_type="inspection")

        # Simulate some work
        for _ in range(10):
            m1.increment_cycle()
        for _ in range(3):
            m1.report_defect()

        for _ in range(5):
            m2.increment_cycle()

        factory.add_machine(m1)
        factory.add_machine(m2)

        stats = factory.get_statistics()

        assert stats["total_machines"] == 2
        assert stats["total_cycles"] == 15
        assert stats["total_defects"] == 3
        assert stats["overall_defect_rate"] == 3 / 15

    def test_to_dict(self):
        """Test factory to dictionary."""
        factory = FactoryState(factory_id="Factory_01")

        m1 = MachineState(machine_id="M001", machine_type="assembly")
        factory.add_machine(m1)

        data = factory.to_dict()

        assert data["factory_id"] == "Factory_01"
        assert "machines" in data
        assert "M001" in data["machines"]
        assert "statistics" in data
