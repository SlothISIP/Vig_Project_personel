import React from 'react';
import { Card } from '@/components/common/Card';
import { StatusBadge } from '@/components/common/StatusBadge';
import { AlertTriangle, Clock, Wrench } from 'lucide-react';
import { formatPercent, timeAgo } from '@/utils/helpers';
import type { MaintenanceRecommendation } from '@/types';

interface MaintenanceCardProps {
  recommendation: MaintenanceRecommendation;
}

export function MaintenanceCard({ recommendation }: MaintenanceCardProps) {
  const isUrgent = ['high', 'critical'].includes(recommendation.urgency);

  return (
    <Card
      className={isUrgent ? 'border-red-300 bg-red-50' : ''}
      title={`Machine: ${recommendation.machine_id}`}
    >
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <StatusBadge status={recommendation.urgency} />
          <span className="text-sm text-gray-600">{timeAgo(recommendation.timestamp)}</span>
        </div>

        <div className="space-y-3">
          <div className="flex items-start">
            <AlertTriangle className="h-5 w-5 text-yellow-600 mr-2 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-gray-700">Failure Probability</p>
              <p className="text-lg font-semibold">
                {formatPercent(recommendation.failure_probability)}
              </p>
            </div>
          </div>

          {recommendation.remaining_useful_life_hours && (
            <div className="flex items-start">
              <Clock className="h-5 w-5 text-blue-600 mr-2 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-gray-700">Remaining Useful Life</p>
                <p className="text-lg font-semibold">
                  {recommendation.remaining_useful_life_hours.toFixed(0)} hours
                </p>
              </div>
            </div>
          )}

          <div className="flex items-start">
            <Wrench className="h-5 w-5 text-green-600 mr-2 mt-0.5" />
            <div>
              <p className="text-sm font-medium text-gray-700">Recommended Action</p>
              <p className="text-sm text-gray-600">{recommendation.recommended_action}</p>
            </div>
          </div>
        </div>

        {recommendation.contributing_factors.length > 0 && (
          <div>
            <p className="text-sm font-medium text-gray-700 mb-2">Contributing Factors</p>
            <ul className="list-disc list-inside space-y-1">
              {recommendation.contributing_factors.map((factor, idx) => (
                <li key={idx} className="text-sm text-gray-600">
                  {factor}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Card>
  );
}
