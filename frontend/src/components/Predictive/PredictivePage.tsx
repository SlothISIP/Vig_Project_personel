/**
 * Predictive Maintenance Page
 * Displays maintenance recommendations and machine health predictions
 */

import React, { useState, useEffect } from 'react';
import { AlertTriangle, Wrench, TrendingDown, Clock } from 'lucide-react';
import { Card } from '@/components/common/Card';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { Alert } from '@/components/common/Alert';
import { MaintenanceCard } from '@/components/Predictive/MaintenanceCard';
import { apiClient } from '@/services/api';
import { wsService } from '@/services/websocket';
import type { MaintenanceRecommendation } from '@/types';

export function PredictivePage() {
  const [recommendations, setRecommendations] = useState<MaintenanceRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterUrgency, setFilterUrgency] = useState<string>('all');

  // Load recommendations
  useEffect(() => {
    loadRecommendations();

    // Subscribe to real-time updates
    wsService.on('maintenance_update', handleMaintenanceUpdate);

    // Refresh every minute
    const interval = setInterval(loadRecommendations, 60000);

    return () => {
      wsService.off('maintenance_update', handleMaintenanceUpdate);
      clearInterval(interval);
    };
  }, []);

  const loadRecommendations = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getMaintenanceRecommendations();
      setRecommendations(data);
    } catch (err) {
      setError('Failed to load maintenance recommendations');
      console.error('Error loading recommendations:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleMaintenanceUpdate = (data: MaintenanceRecommendation) => {
    setRecommendations((prev) => {
      const index = prev.findIndex((r) => r.machine_id === data.machine_id);
      if (index >= 0) {
        const updated = [...prev];
        updated[index] = data;
        return updated;
      }
      return [...prev, data];
    });
  };

  // Filter recommendations
  const filteredRecommendations = recommendations.filter((rec) => {
    if (filterUrgency === 'all') return true;
    return rec.urgency === filterUrgency;
  });

  // Statistics
  const stats = {
    total: recommendations.length,
    critical: recommendations.filter((r) => r.urgency === 'critical').length,
    high: recommendations.filter((r) => r.urgency === 'high').length,
    medium: recommendations.filter((r) => r.urgency === 'medium').length,
    avgHealth: recommendations.length > 0
      ? recommendations.reduce((sum, r) => sum + r.health_score, 0) / recommendations.length
      : 1.0,
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center gap-2">
          <Wrench className="w-8 h-8 text-primary-600" />
          Predictive Maintenance
        </h1>
        <p className="text-gray-600 mt-2">
          AI-powered maintenance recommendations based on machine health and failure predictions
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert type="error" title="Error" className="mb-6">
          {error}
          <button
            onClick={loadRecommendations}
            className="ml-4 text-sm underline"
          >
            Retry
          </button>
        </Alert>
      )}

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
        <Card>
          <Card.Content>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Total Machines</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.total}</p>
              </div>
              <Wrench className="w-10 h-10 text-gray-400" />
            </div>
          </Card.Content>
        </Card>

        <Card>
          <Card.Content>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Critical Alerts</p>
                <p className="text-3xl font-bold text-red-600 mt-1">{stats.critical}</p>
              </div>
              <AlertTriangle className="w-10 h-10 text-red-400" />
            </div>
          </Card.Content>
        </Card>

        <Card>
          <Card.Content>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">High Priority</p>
                <p className="text-3xl font-bold text-orange-600 mt-1">{stats.high}</p>
              </div>
              <TrendingDown className="w-10 h-10 text-orange-400" />
            </div>
          </Card.Content>
        </Card>

        <Card>
          <Card.Content>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Avg Health Score</p>
                <p className="text-3xl font-bold text-green-600 mt-1">
                  {(stats.avgHealth * 100).toFixed(0)}%
                </p>
              </div>
              <Clock className="w-10 h-10 text-green-400" />
            </div>
          </Card.Content>
        </Card>
      </div>

      {/* Filter Tabs */}
      <div className="mb-6">
        <div className="flex gap-2 flex-wrap">
          {['all', 'critical', 'high', 'medium', 'low', 'none'].map((urgency) => (
            <button
              key={urgency}
              onClick={() => setFilterUrgency(urgency)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                filterUrgency === urgency
                  ? 'bg-primary-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-300'
              }`}
            >
              {urgency.charAt(0).toUpperCase() + urgency.slice(1)}
              {urgency !== 'all' && (
                <span className="ml-2 px-2 py-0.5 rounded-full text-xs bg-gray-200 text-gray-700">
                  {recommendations.filter((r) => r.urgency === urgency).length}
                </span>
              )}
              {urgency === 'all' && (
                <span className="ml-2 px-2 py-0.5 rounded-full text-xs bg-gray-200 text-gray-700">
                  {recommendations.length}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Recommendations Grid */}
      {filteredRecommendations.length === 0 ? (
        <Card>
          <Card.Content>
            <div className="text-center py-12">
              <Wrench className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-600 text-lg">
                {filterUrgency === 'all'
                  ? 'No maintenance recommendations available'
                  : `No ${filterUrgency} urgency recommendations`}
              </p>
              <p className="text-gray-500 text-sm mt-2">
                All machines are operating within normal parameters
              </p>
            </div>
          </Card.Content>
        </Card>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {filteredRecommendations
            .sort((a, b) => {
              // Sort by urgency: critical > high > medium > low > none
              const urgencyOrder = { critical: 5, high: 4, medium: 3, low: 2, none: 1 };
              const orderA = urgencyOrder[a.urgency as keyof typeof urgencyOrder] || 0;
              const orderB = urgencyOrder[b.urgency as keyof typeof urgencyOrder] || 0;
              if (orderA !== orderB) return orderB - orderA;
              // Then by failure probability
              return b.failure_probability - a.failure_probability;
            })
            .map((recommendation) => (
              <MaintenanceCard
                key={recommendation.machine_id}
                recommendation={recommendation}
              />
            ))}
        </div>
      )}
    </div>
  );
}
