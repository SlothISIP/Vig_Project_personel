import React from 'react';
import { Card } from '@/components/common/Card';
import { formatDate, formatDuration } from '@/utils/helpers';
import type { Schedule, ScheduleAssignment } from '@/types';

interface ScheduleTimelineProps {
  schedule: Schedule;
}

export function ScheduleTimeline({ schedule }: ScheduleTimelineProps) {
  const groupedByMachine = schedule.assignments.reduce((acc, assignment) => {
    if (!acc[assignment.machine_id]) {
      acc[assignment.machine_id] = [];
    }
    acc[assignment.machine_id].push(assignment);
    return acc;
  }, {} as Record<string, ScheduleAssignment[]>);

  return (
    <Card title="Schedule Timeline" subtitle={`Makespan: ${schedule.statistics.makespan_minutes} minutes`}>
      <div className="space-y-6">
        {Object.entries(groupedByMachine).map(([machineId, assignments]) => (
          <div key={machineId}>
            <h4 className="font-semibold text-gray-900 mb-3">{machineId}</h4>
            <div className="space-y-2">
              {assignments
                .sort((a, b) => new Date(a.start_time).getTime() - new Date(b.start_time).getTime())
                .map((assignment, idx) => (
                  <div
                    key={idx}
                    className="flex items-center p-3 bg-gray-50 rounded-lg border border-gray-200"
                  >
                    <div className="flex-1">
                      <p className="font-medium text-gray-900">{assignment.job_id}</p>
                      <p className="text-sm text-gray-600">{assignment.task_id}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-medium text-gray-900">
                        {formatDate(assignment.start_time, 'HH:mm')} -{' '}
                        {formatDate(assignment.end_time, 'HH:mm')}
                      </p>
                      <p className="text-xs text-gray-600">
                        {formatDuration(
                          (new Date(assignment.end_time).getTime() -
                            new Date(assignment.start_time).getTime()) /
                            60000
                        )}
                      </p>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
