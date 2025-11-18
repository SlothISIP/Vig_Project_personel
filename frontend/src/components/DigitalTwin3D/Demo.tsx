/**
 * 3D Visualization Demo Component
 * Demonstrates the 3D factory visualization with mock data
 */

import React, { useState, useEffect } from 'react';
import { FactoryScene } from './FactoryScene';
import type { MachineState } from '@/types';

/**
 * Demo component with mock data for testing 3D visualization
 */
export function DigitalTwin3DDemo() {
  const [machines, setMachines] = useState<MachineState[]>([]);
  const [selectedMachine, setSelectedMachine] = useState<string | null>(null);

  // Initialize mock machines
  useEffect(() => {
    const mockMachines: MachineState[] = [
      {
        machine_id: 'M001',
        machine_name: 'Assembly Station 1',
        machine_type: 'Assembly',
        status: 'running',
        health_score: 0.92,
        temperature: 68.5,
        vibration: 1.8,
        pressure: 95.2,
        speed: 850,
        defect_rate: 0.02,
        cycle_count: 15420,
        defect_count: 308,
        last_maintenance: '2024-01-10T14:30:00Z',
      },
      {
        machine_id: 'M002',
        machine_name: 'Quality Check',
        machine_type: 'Inspection',
        status: 'running',
        health_score: 0.88,
        temperature: 72.1,
        vibration: 2.3,
        pressure: 92.8,
        speed: 780,
        defect_rate: 0.05,
        cycle_count: 14850,
        defect_count: 742,
        last_maintenance: '2024-01-08T09:15:00Z',
      },
      {
        machine_id: 'M003',
        machine_name: 'Packaging Unit',
        machine_type: 'Packaging',
        status: 'warning',
        health_score: 0.65,
        temperature: 78.9,
        vibration: 3.5,
        pressure: 88.3,
        speed: 650,
        defect_rate: 0.08,
        cycle_count: 13200,
        defect_count: 1056,
        last_maintenance: '2023-12-28T16:45:00Z',
      },
    ];

    setMachines(mockMachines);

    // Simulate real-time updates
    const interval = setInterval(() => {
      setMachines((prev) =>
        prev.map((machine) => ({
          ...machine,
          health_score: Math.max(0.5, Math.min(1, machine.health_score + (Math.random() - 0.5) * 0.05)),
          temperature: machine.temperature + (Math.random() - 0.5) * 2,
          vibration: Math.max(0, machine.vibration + (Math.random() - 0.5) * 0.3),
          cycle_count: machine.cycle_count + Math.floor(Math.random() * 3),
        }))
      );
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const handleMachineClick = (machineId: string) => {
    setSelectedMachine(machineId);
    console.log('Machine clicked:', machineId);
  };

  const selectedMachineData = machines.find((m) => m.machine_id === selectedMachine);

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900">
          3D Digital Twin Demo
        </h1>
        <p className="text-sm text-gray-600 mt-1">
          Interactive 3D factory visualization with real-time updates
        </p>
      </div>

      {/* Content */}
      <div className="flex-1 flex">
        {/* 3D View */}
        <div className="flex-1">
          <FactoryScene
            machines={machines}
            onMachineClick={handleMachineClick}
            showProductFlow={true}
          />
        </div>

        {/* Info Panel */}
        <div className="w-80 bg-white border-l border-gray-200 p-6 overflow-y-auto">
          <h2 className="text-lg font-semibold mb-4">Machine Status</h2>

          {selectedMachineData ? (
            <div className="space-y-4">
              <div>
                <h3 className="text-base font-medium">{selectedMachineData.machine_name}</h3>
                <p className="text-sm text-gray-600">{selectedMachineData.machine_id}</p>
              </div>

              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Status:</span>
                  <span className={`text-sm font-medium ${
                    selectedMachineData.status === 'running' ? 'text-green-600' :
                    selectedMachineData.status === 'warning' ? 'text-yellow-600' :
                    'text-red-600'
                  }`}>
                    {selectedMachineData.status.toUpperCase()}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Health:</span>
                  <span className="text-sm font-medium">
                    {(selectedMachineData.health_score * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Temperature:</span>
                  <span className="text-sm font-medium">
                    {selectedMachineData.temperature?.toFixed(1)}°C
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Vibration:</span>
                  <span className="text-sm font-medium">
                    {selectedMachineData.vibration?.toFixed(2)} mm/s
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Cycles:</span>
                  <span className="text-sm font-medium">
                    {selectedMachineData.cycle_count.toLocaleString()}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Defect Rate:</span>
                  <span className="text-sm font-medium">
                    {(selectedMachineData.defect_rate * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              <button
                onClick={() => setSelectedMachine(null)}
                className="w-full btn btn-secondary mt-4"
              >
                Clear Selection
              </button>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              <p className="text-sm">Click on a machine in the 3D view to see details</p>
            </div>
          )}

          {/* Machine List */}
          <div className="mt-8">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">All Machines</h3>
            <div className="space-y-2">
              {machines.map((machine) => (
                <button
                  key={machine.machine_id}
                  onClick={() => handleMachineClick(machine.machine_id)}
                  className={`w-full text-left p-3 rounded-lg border transition-all ${
                    selectedMachine === machine.machine_id
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300 bg-white'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium">{machine.machine_name}</div>
                      <div className="text-xs text-gray-600">{machine.machine_id}</div>
                    </div>
                    <div className={`w-2 h-2 rounded-full ${
                      machine.status === 'running' ? 'bg-green-500' :
                      machine.status === 'warning' ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`} />
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Instructions */}
          <div className="mt-8 p-4 bg-blue-50 rounded-lg">
            <h4 className="text-sm font-semibold text-blue-900 mb-2">Controls</h4>
            <ul className="text-xs text-blue-800 space-y-1">
              <li>• <strong>Rotate:</strong> Left click + drag</li>
              <li>• <strong>Pan:</strong> Right click + drag</li>
              <li>• <strong>Zoom:</strong> Scroll wheel</li>
              <li>• <strong>Select:</strong> Click on machines</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DigitalTwin3DDemo;
