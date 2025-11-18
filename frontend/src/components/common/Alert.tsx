import React from 'react';
import { AlertCircle, AlertTriangle, Info, CheckCircle, X } from 'lucide-react';
import { cn } from '@/utils/helpers';

interface AlertProps {
  type: 'info' | 'warning' | 'error' | 'success';
  message: string;
  onClose?: () => void;
  className?: string;
}

const alertStyles = {
  info: {
    container: 'bg-blue-50 border-blue-200 text-blue-800',
    icon: <Info className="h-5 w-5 text-blue-600" />,
  },
  warning: {
    container: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    icon: <AlertTriangle className="h-5 w-5 text-yellow-600" />,
  },
  error: {
    container: 'bg-red-50 border-red-200 text-red-800',
    icon: <AlertCircle className="h-5 w-5 text-red-600" />,
  },
  success: {
    container: 'bg-green-50 border-green-200 text-green-800',
    icon: <CheckCircle className="h-5 w-5 text-green-600" />,
  },
};

export function Alert({ type, message, onClose, className }: AlertProps) {
  const style = alertStyles[type];

  return (
    <div className={cn('border rounded-lg p-4 flex items-start', style.container, className)}>
      <div className="flex-shrink-0">{style.icon}</div>
      <div className="ml-3 flex-1">
        <p className="text-sm font-medium">{message}</p>
      </div>
      {onClose && (
        <button
          onClick={onClose}
          className="ml-3 flex-shrink-0 hover:opacity-70 transition-opacity"
        >
          <X className="h-5 w-5" />
        </button>
      )}
    </div>
  );
}
