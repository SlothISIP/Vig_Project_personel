// API Types

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  message?: string;
  error?: string;
}

// Vision AI Types
export interface PredictionResult {
  predicted_class: number;
  confidence: number;
  defect_type: string;
  inference_time_ms: number;
  probabilities?: number[];
}

// Digital Twin Types
export interface MachineState {
  machine_id: string;
  machine_type: string;
  status: 'idle' | 'running' | 'warning' | 'error' | 'maintenance';
  health_score: number;
  cycle_count: number;
  defect_count: number;
  defect_rate: number;
  last_maintenance?: string;
  updated_at: string;
}

export interface FactoryState {
  factory_id: string;
  machines: Record<string, MachineState>;
  statistics: {
    total_machines: number;
    active_machines: number;
    overall_health: number;
    total_cycles: number;
    total_defects: number;
    overall_defect_rate: number;
  };
}

export interface SensorReading {
  sensor_id: string;
  sensor_type: string;
  value: number;
  unit: string;
  timestamp: string;
  quality: number;
  anomaly_score: number;
}

// Predictive Maintenance Types
export interface MaintenanceRecommendation {
  machine_id: string;
  timestamp: string;
  failure_probability: number;
  failure_risk_level: string;
  remaining_useful_life_hours: number | null;
  health_score: number;
  urgency: 'none' | 'low' | 'medium' | 'high' | 'critical';
  recommended_action: string;
  estimated_downtime_hours: number;
  maintenance_window?: [string, string];
  confidence: number;
  contributing_factors: string[];
}

// Scheduling Types
export interface Job {
  job_id: string;
  product_type: string;
  priority: number;
  due_date?: string;
  release_date?: string;
  tasks: Task[];
}

export interface Task {
  task_id: string;
  task_type: string;
  duration: number;
  status: 'pending' | 'scheduled' | 'in_progress' | 'completed' | 'failed';
  assigned_machine_id?: string;
  scheduled_start?: string;
  scheduled_end?: string;
}

export interface Schedule {
  schedule_id: string;
  created_at: string;
  assignments: ScheduleAssignment[];
  statistics: {
    total_assignments: number;
    makespan_minutes: number;
    num_machines: number;
    average_utilization: number;
    has_conflicts: boolean;
  };
}

export interface ScheduleAssignment {
  task_id: string;
  job_id: string;
  machine_id: string;
  start_time: string;
  end_time: string;
}

// Dashboard Types
export interface DashboardStats {
  total_machines: number;
  active_machines: number;
  total_products: number;
  defect_rate: number;
  overall_efficiency: number;
  alerts: Alert[];
}

export interface Alert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
  machine_id?: string;
}

// Chart Data Types
export interface TimeSeriesDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    borderColor?: string;
    backgroundColor?: string;
  }[];
}
