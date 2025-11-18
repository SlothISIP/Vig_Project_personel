import { format, formatDistanceToNow } from 'date-fns';
import clsx, { ClassValue } from 'clsx';

// Tailwind class name combiner
export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

// Date formatting
export function formatDate(date: string | Date, formatStr: string = 'PPp'): string {
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    return format(dateObj, formatStr);
  } catch {
    return 'Invalid date';
  }
}

export function timeAgo(date: string | Date): string {
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date;
    return formatDistanceToNow(dateObj, { addSuffix: true });
  } catch {
    return 'Unknown';
  }
}

// Number formatting
export function formatNumber(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}

export function formatPercent(value: number, decimals: number = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function formatDuration(minutes: number): string {
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;

  if (hours > 0) {
    return `${hours}h ${mins}m`;
  }
  return `${mins}m`;
}

// Status helpers
export function getStatusColor(
  status: string
): 'green' | 'yellow' | 'red' | 'gray' | 'blue' {
  const statusMap: Record<string, 'green' | 'yellow' | 'red' | 'gray' | 'blue'> = {
    idle: 'gray',
    running: 'green',
    warning: 'yellow',
    error: 'red',
    maintenance: 'blue',
    critical: 'red',
    high: 'red',
    medium: 'yellow',
    low: 'green',
    none: 'gray',
  };

  return statusMap[status.toLowerCase()] || 'gray';
}

export function getHealthColor(health: number): string {
  if (health >= 0.8) return 'text-green-600';
  if (health >= 0.5) return 'text-yellow-600';
  return 'text-red-600';
}

// Data transformations
export function groupBy<T>(array: T[], key: keyof T): Record<string, T[]> {
  return array.reduce((result, item) => {
    const groupKey = String(item[key]);
    if (!result[groupKey]) {
      result[groupKey] = [];
    }
    result[groupKey].push(item);
    return result;
  }, {} as Record<string, T[]>);
}

export function sortBy<T>(array: T[], key: keyof T, order: 'asc' | 'desc' = 'asc'): T[] {
  return [...array].sort((a, b) => {
    const aVal = a[key];
    const bVal = b[key];

    if (aVal < bVal) return order === 'asc' ? -1 : 1;
    if (aVal > bVal) return order === 'asc' ? 1 : -1;
    return 0;
  });
}

// Error handling
export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  return 'An unknown error occurred';
}

// Debounce
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
