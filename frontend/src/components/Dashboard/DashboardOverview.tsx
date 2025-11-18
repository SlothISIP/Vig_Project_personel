import React, { useState, useEffect } from 'react';
import { Activity, AlertTriangle, TrendingUp, Package } from 'lucide-react';
import { StatCard } from '@/components/common/Card';
import { Alert } from '@/components/common/Alert';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { apiClient } from '@/services/api';
import { formatPercent } from '@/utils/helpers';
import type { DashboardStats } from '@/types';

export function DashboardOverview() {
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadStats();
    const interval = setInterval(loadStats, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const loadStats = async () => {
    try {
      const data = await apiClient.getDashboardStats();
      setStats(data);
      setError(null);
    } catch (err) {
      setError('Failed to load dashboard statistics');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <Alert type="error" message={error} />;
  }

  return (
    <div className="space-y-6">
      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Machines"
          value={stats?.total_machines || 0}
          subtitle={`${stats?.active_machines || 0} active`}
          icon={<Activity className="h-8 w-8" />}
        />

        <StatCard
          title="Total Products"
          value={stats?.total_products || 0}
          icon={<Package className="h-8 w-8" />}
        />

        <StatCard
          title="Defect Rate"
          value={formatPercent(stats?.defect_rate || 0)}
          trend={{
            value: 2.5,
            isPositive: false,
          }}
          icon={<AlertTriangle className="h-8 w-8" />}
        />

        <StatCard
          title="Overall Efficiency"
          value={formatPercent(stats?.overall_efficiency || 0)}
          trend={{
            value: 5.2,
            isPositive: true,
          }}
          icon={<TrendingUp className="h-8 w-8" />}
        />
      </div>

      {/* Alerts */}
      {stats?.alerts && stats.alerts.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-lg font-semibold">Recent Alerts</h3>
          {stats.alerts.slice(0, 5).map((alert) => (
            <Alert key={alert.id} type={alert.type} message={alert.message} />
          ))}
        </div>
      )}
    </div>
  );
}
