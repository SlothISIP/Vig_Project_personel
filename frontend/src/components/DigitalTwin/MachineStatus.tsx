import React from 'react';
import { Card } from '@/components/common/Card';
import { StatusBadge } from '@/components/common/StatusBadge';
import { formatPercent, formatNumber } from '@/utils/helpers';
import type { MachineState } from '@/types';

interface MachineStatusProps {
  machine: MachineState;
}

export function MachineStatus({ machine }: MachineStatusProps) {
  return (
    <Card className="hover:shadow-lg transition-shadow">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h4 className="text-lg font-semibold">{machine.machine_id}</h4>
            <p className="text-sm text-gray-600">{machine.machine_type}</p>
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
        </div>
      </div>
    </Card>
  );
}
