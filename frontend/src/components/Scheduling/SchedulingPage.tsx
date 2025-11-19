/**
 * Production Scheduling Page
 * Displays production schedules, job status, and machine assignments
 */

import React, { useState, useEffect } from 'react';
import { Calendar, TrendingUp, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import { Card } from '@/components/common/Card';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { Alert } from '@/components/common/Alert';
import { ScheduleTimeline } from '@/components/Scheduling/ScheduleTimeline';
import { apiClient } from '@/services/api';
import { wsService } from '@/services/websocket';
import type { Schedule } from '@/types';

export function SchedulingPage() {
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMachine, setSelectedMachine] = useState<string>('all');

  // Load schedules
  useEffect(() => {
    loadSchedules();

    // Subscribe to real-time updates
    wsService.on('schedule_update', handleScheduleUpdate);

    // Refresh every 30 seconds
    const interval = setInterval(loadSchedules, 30000);

    return () => {
      wsService.off('schedule_update', handleScheduleUpdate);
      clearInterval(interval);
    };
  }, []);

  const loadSchedules = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await apiClient.getSchedules();
      setSchedules(data);
    } catch (err) {
      setError('Failed to load production schedules');
      console.error('Error loading schedules:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleScheduleUpdate = (data: Schedule) => {
    setSchedules((prev) => {
      const index = prev.findIndex((s) => s.schedule_id === data.schedule_id);
      if (index >= 0) {
        const updated = [...prev];
        updated[index] = data;
        return updated;
      }
      return [...prev, data];
    });
  };

  // Filter schedules by machine
  const filteredSchedules = schedules.filter((schedule) => {
    if (selectedMachine === 'all') return true;
    return schedule.assignments.some((a) => a.machine_id === selectedMachine);
  });

  // Get unique machines
  const machines = Array.from(
    new Set(
      schedules.flatMap((s) => s.assignments.map((a) => a.machine_id))
    )
  ).sort();

  // Statistics
  const stats = {
    totalJobs: schedules.reduce((sum, s) => sum + s.jobs.length, 0),
    completedJobs: schedules.reduce(
      (sum, s) => sum + s.jobs.filter((j) => j.status === 'completed').length,
      0
    ),
    activeJobs: schedules.reduce(
      (sum, s) => sum + s.jobs.filter((j) => j.status === 'in_progress').length,
      0
    ),
    pendingJobs: schedules.reduce(
      (sum, s) => sum + s.jobs.filter((j) => j.status === 'pending').length,
      0
    ),
    avgUtilization: schedules.length > 0
      ? schedules.reduce((sum, s) => sum + (s.utilization || 0), 0) / schedules.length
      : 0,
    overdueJobs: schedules.reduce(
      (sum, s) => sum + s.jobs.filter((j) => j.is_overdue).length,
      0
    ),
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
          <Calendar className="w-8 h-8 text-primary-600" />
          Production Scheduling
        </h1>
        <p className="text-gray-600 mt-2">
          Optimized production schedules with real-time job tracking and machine assignments
        </p>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert type="error" title="Error" className="mb-6">
          {error}
          <button
            onClick={loadSchedules}
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
                <p className="text-sm text-gray-600">Total Jobs</p>
                <p className="text-3xl font-bold text-gray-900 mt-1">{stats.totalJobs}</p>
              </div>
              <Calendar className="w-10 h-10 text-gray-400" />
            </div>
          </Card.Content>
        </Card>

        <Card>
          <Card.Content>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Completed</p>
                <p className="text-3xl font-bold text-green-600 mt-1">{stats.completedJobs}</p>
              </div>
              <CheckCircle className="w-10 h-10 text-green-400" />
            </div>
          </Card.Content>
        </Card>

        <Card>
          <Card.Content>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">In Progress</p>
                <p className="text-3xl font-bold text-blue-600 mt-1">{stats.activeJobs}</p>
              </div>
              <Clock className="w-10 h-10 text-blue-400" />
            </div>
          </Card.Content>
        </Card>

        <Card>
          <Card.Content>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Overdue</p>
                <p className="text-3xl font-bold text-red-600 mt-1">{stats.overdueJobs}</p>
              </div>
              <AlertCircle className="w-10 h-10 text-red-400" />
            </div>
          </Card.Content>
        </Card>
      </div>

      {/* Additional Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
        <Card>
          <Card.Content>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Pending Jobs</p>
                <p className="text-2xl font-bold text-gray-700 mt-1">{stats.pendingJobs}</p>
              </div>
              <p className="text-xs text-gray-500">
                Waiting to be scheduled
              </p>
            </div>
          </Card.Content>
        </Card>

        <Card>
          <Card.Content>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-600">Avg Utilization</p>
                <p className="text-2xl font-bold text-primary-600 mt-1">
                  {(stats.avgUtilization * 100).toFixed(1)}%
                </p>
              </div>
              <TrendingUp className="w-10 h-10 text-primary-400" />
            </div>
          </Card.Content>
        </Card>
      </div>

      {/* Machine Filter */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Filter by Machine
        </label>
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => setSelectedMachine('all')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              selectedMachine === 'all'
                ? 'bg-primary-600 text-white'
                : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-300'
            }`}
          >
            All Machines
          </button>
          {machines.map((machineId) => (
            <button
              key={machineId}
              onClick={() => setSelectedMachine(machineId)}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                selectedMachine === machineId
                  ? 'bg-primary-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-300'
              }`}
            >
              {machineId}
            </button>
          ))}
        </div>
      </div>

      {/* Schedules List */}
      {filteredSchedules.length === 0 ? (
        <Card>
          <Card.Content>
            <div className="text-center py-12">
              <Calendar className="w-16 h-16 text-gray-300 mx-auto mb-4" />
              <p className="text-gray-600 text-lg">
                {selectedMachine === 'all'
                  ? 'No production schedules available'
                  : `No schedules for machine ${selectedMachine}`}
              </p>
              <p className="text-gray-500 text-sm mt-2">
                Create a new schedule or wait for jobs to be assigned
              </p>
            </div>
          </Card.Content>
        </Card>
      ) : (
        <div className="space-y-6">
          {filteredSchedules.map((schedule) => (
            <Card key={schedule.schedule_id}>
              <Card.Header>
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">
                      Schedule: {schedule.schedule_id}
                    </h3>
                    <p className="text-sm text-gray-600 mt-1">
                      Created: {new Date(schedule.created_at).toLocaleString()}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">Makespan</p>
                    <p className="text-lg font-bold text-primary-600">
                      {schedule.makespan} min
                    </p>
                  </div>
                </div>
              </Card.Header>
              <Card.Content>
                <ScheduleTimeline schedule={schedule} />

                {/* Job Details */}
                <div className="mt-6">
                  <h4 className="text-sm font-semibold text-gray-700 mb-3">
                    Job Details ({schedule.jobs.length} jobs)
                  </h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {schedule.jobs.map((job) => (
                      <div
                        key={job.job_id}
                        className={`p-4 rounded-lg border-2 ${
                          job.is_overdue
                            ? 'border-red-300 bg-red-50'
                            : job.status === 'completed'
                            ? 'border-green-300 bg-green-50'
                            : job.status === 'in_progress'
                            ? 'border-blue-300 bg-blue-50'
                            : 'border-gray-300 bg-gray-50'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-medium text-gray-900">{job.job_id}</span>
                          <span
                            className={`px-2 py-1 rounded text-xs font-medium ${
                              job.status === 'completed'
                                ? 'bg-green-100 text-green-800'
                                : job.status === 'in_progress'
                                ? 'bg-blue-100 text-blue-800'
                                : 'bg-gray-100 text-gray-800'
                            }`}
                          >
                            {job.status}
                          </span>
                        </div>
                        <div className="text-sm text-gray-600 space-y-1">
                          <div>Priority: {job.priority}</div>
                          <div>Duration: {job.duration} min</div>
                          {job.deadline && (
                            <div className={job.is_overdue ? 'text-red-600 font-medium' : ''}>
                              Deadline: {new Date(job.deadline).toLocaleString()}
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </Card.Content>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
