import React from 'react';
import { Card } from '@/components/common/Card';
import { StatusBadge } from '@/components/common/StatusBadge';
import { formatPercent, formatNumber } from '@/utils/helpers';
import type { MachineState } from '@/types';

interface MachineStatusProps {
  machine: MachineState;
  compact?: boolean;
}

export function MachineStatus({ machine, compact = false }: MachineStatusProps) {
  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-base font-semibold">{machine.machine_name}</h4>
            <p className="text-xs text-gray-600">{machine.machine_id}</p>
          </div>
          <StatusBadge status={machine.status} />
        </div>

        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <p className="text-xs text-gray-600">Health</p>
            <p className="font-semibold">{formatPercent(machine.health_score)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-600">Defects</p>
            <p className="font-semibold">{formatPercent(machine.defect_rate)}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-lg font-semibold">{machine.machine_name}</h4>
          <p className="text-sm text-gray-600">{machine.machine_id} - {machine.machine_type}</p>
        </div>
        <StatusBadge status={machine.status} />
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-gray-600">Health Score</p>
          <p className="text-2xl font-semibold">{formatPercent(machine.health_score)}</p>
        </div>

        <div>
          <p className="text-sm text-gray-600">Defect Rate</p>
          <p className="text-2xl font-semibold">{formatPercent(machine.defect_rate)}</p>
        </div>

        <div>
          <p className="text-sm text-gray-600">Cycle Count</p>
          <p className="text-lg">{formatNumber(machine.cycle_count, 0)}</p>
        </div>

        <div>
          <p className="text-sm text-gray-600">Defect Count</p>
          <p className="text-lg">{formatNumber(machine.defect_count, 0)}</p>
        </div>

        {machine.temperature !== undefined && (
          <div>
            <p className="text-sm text-gray-600">Temperature</p>
            <p className="text-lg">{machine.temperature.toFixed(1)}°C</p>
          </div>
        )}

        {machine.vibration !== undefined && (
          <div>
            <p className="text-sm text-gray-600">Vibration</p>
            <p className="text-lg">{machine.vibration.toFixed(2)} mm/s</p>
          </div>
        )}
      </div>
    </div>
  );
}
