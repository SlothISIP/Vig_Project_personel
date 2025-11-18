import axios, { AxiosInstance } from 'axios';
import type {
  ApiResponse,
  PredictionResult,
  FactoryState,
  MaintenanceRecommendation,
  Schedule,
  Job,
  DashboardStats,
} from '@/types';

class ApiClient {
  private client: AxiosInstance;

  constructor(baseURL: string = '/api/v1') {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        return Promise.reject(error);
      }
    );
  }

  // Vision AI
  async predict(image: File): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('file', image);

    const response = await this.client.post<PredictionResult>(
      '/predict',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }

  async predictBatch(images: File[]): Promise<PredictionResult[]> {
    const formData = new FormData();
    images.forEach((image) => {
      formData.append('files', image);
    });

    const response = await this.client.post<PredictionResult[]>(
      '/predict/batch',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return response.data;
  }

  // Digital Twin
  async getFactoryState(factoryId: string = 'Factory_01'): Promise<FactoryState> {
    const response = await this.client.get<FactoryState>(
      `/digital-twin/factory/${factoryId}`
    );
    return response.data;
  }

  async getMachineState(machineId: string): Promise<any> {
    const response = await this.client.get(`/digital-twin/machine/${machineId}`);
    return response.data;
  }

  // Predictive Maintenance
  async getMaintenanceRecommendation(
    machineId: string
  ): Promise<MaintenanceRecommendation> {
    const response = await this.client.get<MaintenanceRecommendation>(
      `/predictive/maintenance/${machineId}`
    );
    return response.data;
  }

  async getAllMaintenanceRecommendations(): Promise<MaintenanceRecommendation[]> {
    const response = await this.client.get<MaintenanceRecommendation[]>(
      '/predictive/maintenance/all'
    );
    return response.data;
  }

  // Scheduling
  async getCurrentSchedule(): Promise<Schedule> {
    const response = await this.client.get<Schedule>('/scheduling/current');
    return response.data;
  }

  async createSchedule(jobs: Job[]): Promise<Schedule> {
    const response = await this.client.post<Schedule>('/scheduling/schedule', {
      jobs,
    });
    return response.data;
  }

  async getJobStatus(jobId: string): Promise<any> {
    const response = await this.client.get(`/scheduling/job/${jobId}`);
    return response.data;
  }

  // Dashboard
  async getDashboardStats(): Promise<DashboardStats> {
    const response = await this.client.get<DashboardStats>('/dashboard/stats');
    return response.data;
  }

  // Health Check
  async healthCheck(): Promise<{ status: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }
}

export const apiClient = new ApiClient();
export default apiClient;
