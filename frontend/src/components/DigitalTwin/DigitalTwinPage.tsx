/**
 * Digital Twin Page
 * Combines 3D visualization with real-time machine data and controls
 */

import React, { useState, useEffect, Suspense } from 'react';
import { FactoryScene, FactorySceneLoader } from '@/components/DigitalTwin3D';
import { MachineStatus } from './MachineStatus';
import { Card } from '@/components/common/Card';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { Alert } from '@/components/common/Alert';
import { apiClient } from '@/services/api';
import { wsService } from '@/services/websocket';
import type { FactoryState, MachineState } from '@/types';

export function DigitalTwinPage() {
  const [factoryState, setFactoryState] = useState<FactoryState | null>(null);
  const [selectedMachine, setSelectedMachine] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'3d' | '2d' | 'split'>('split');

  // Load factory state
  useEffect(() => {
    loadFactoryState();

    // Subscribe to real-time updates
    wsService.on('factory_update', handleFactoryUpdate);
    wsService.on('machine_update', handleMachineUpdate);

    return () => {
      wsService.off('factory_update', handleFactoryUpdate);
      wsService.off('machine_update', handleMachineUpdate);
    };
  }, []);

  const loadFactoryState = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getFactoryState('factory-1');
      setFactoryState(data);
    } catch (err) {
      setError('Failed to load factory state');
      console.error('Error loading factory state:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFactoryUpdate = (data: FactoryState) => {
    setFactoryState(data);
  };

  const handleMachineUpdate = (data: { machine_id: string; updates: Partial<MachineState> }) => {
    if (!factoryState) return;

    setFactoryState({
      ...factoryState,
      machines: factoryState.machines.map((machine) =>
        machine.machine_id === data.machine_id
          ? { ...machine, ...data.updates }
          : machine
      ),
    });
  };

  const handleMachineClick = (machineId: string) => {
    setSelectedMachine(machineId);
  };

  const selectedMachineData = factoryState?.machines.find(
    (m) => m.machine_id === selectedMachine
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert type="error" title="Error Loading Factory">
        {error}
      </Alert>
    );
  }

  if (!factoryState) {
    return (
      <Alert type="warning" title="No Data">
        No factory data available
      </Alert>
    );
  }

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Digital Twin Factory</h1>
            <p className="text-sm text-gray-600 mt-1">
              Factory ID: {factoryState.factory_id} | Status: {factoryState.status}
            </p>
          </div>

          {/* View mode selector */}
          <div className="flex gap-2">
            <button
              onClick={() => setViewMode('3d')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                viewMode === '3d'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              3D View
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                viewMode === 'split'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              Split View
            </button>
            <button
              onClick={() => setViewMode('2d')}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                viewMode === '2d'
                  ? 'bg-primary-600 text-white'
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              2D View
            </button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* 3D View */}
        {(viewMode === '3d' || viewMode === 'split') && (
          <div className={viewMode === 'split' ? 'w-2/3' : 'w-full'}>
            <Suspense fallback={<FactorySceneLoader />}>
              <FactoryScene
                machines={factoryState.machines}
                onMachineClick={handleMachineClick}
                showProductFlow={true}
              />
            </Suspense>
          </div>
        )}

        {/* 2D View / Details Panel */}
        {(viewMode === '2d' || viewMode === 'split') && (
          <div
            className={`${
              viewMode === 'split' ? 'w-1/3' : 'w-full'
            } bg-gray-50 overflow-y-auto`}
          >
            <div className="p-6 space-y-6">
              {/* Factory metrics */}
              <Card>
                <Card.Header>
                  <h2 className="text-lg font-semibold">Factory Metrics</h2>
                </Card.Header>
                <Card.Content>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">Total Products</div>
                      <div className="text-2xl font-bold text-gray-900">
                        {factoryState.total_products_produced}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Defects</div>
                      <div className="text-2xl font-bold text-red-600">
                        {factoryState.total_defects_detected}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">OEE</div>
                      <div className="text-2xl font-bold text-green-600">
                        {(factoryState.overall_oee * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Uptime</div>
                      <div className="text-2xl font-bold text-blue-600">
                        {(factoryState.uptime_percentage * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                </Card.Content>
              </Card>

              {/* Selected machine details */}
              {selectedMachineData && (
                <Card>
                  <Card.Header>
                    <h2 className="text-lg font-semibold">
                      Selected: {selectedMachineData.machine_name}
                    </h2>
                    <button
                      onClick={() => setSelectedMachine(null)}
                      className="text-sm text-gray-600 hover:text-gray-900"
                    >
                      Clear
                    </button>
                  </Card.Header>
                  <Card.Content>
                    <MachineStatus machine={selectedMachineData} />
                  </Card.Content>
                </Card>
              )}

              {/* All machines list */}
              <div>
                <h2 className="text-lg font-semibold text-gray-900 mb-4">
                  All Machines ({factoryState.machines.length})
                </h2>
                <div className="space-y-4">
                  {factoryState.machines.map((machine) => (
                    <div
                      key={machine.machine_id}
                      onClick={() => handleMachineClick(machine.machine_id)}
                      className={`cursor-pointer transition-all ${
                        selectedMachine === machine.machine_id
                          ? 'ring-2 ring-primary-500'
                          : ''
                      }`}
                    >
                      <Card>
                        <Card.Content>
                          <MachineStatus machine={machine} compact />
                        </Card.Content>
                      </Card>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
