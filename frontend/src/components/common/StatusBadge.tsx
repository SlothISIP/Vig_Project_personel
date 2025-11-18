import React from 'react';
import { cn, getStatusColor } from '@/utils/helpers';

interface StatusBadgeProps {
  status: string;
  className?: string;
}

const colorClasses = {
  green: 'bg-green-100 text-green-800 border-green-200',
  yellow: 'bg-yellow-100 text-yellow-800 border-yellow-200',
  red: 'bg-red-100 text-red-800 border-red-200',
  gray: 'bg-gray-100 text-gray-800 border-gray-200',
  blue: 'bg-blue-100 text-blue-800 border-blue-200',
};

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const color = getStatusColor(status);
  const colorClass = colorClasses[color];

  return (
    <span
      className={cn(
        'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border',
        colorClass,
        className
      )}
    >
      <span className={cn('w-2 h-2 rounded-full mr-1.5', `bg-${color}-600`)} />
      {status.charAt(0).toUpperCase() + status.slice(1)}
    </span>
  );
}
