import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({ baseURL: API_BASE, timeout: 120_000 });

export interface AnalysisStatus {
  analysis_id: string;
  status: string;
  truth_label?: string;
  truth_score?: number;
  truth_confidence?: number;
  current_step?: string;
  completed_steps?: string[];
  progress?: {
    current_step?: string;
    completed_steps?: string[];
    stage_statuses?: Record<string, 'pending' | 'running' | 'completed' | 'partial' | 'failed' | 'unavailable' | 'skipped'>;
    total_steps?: number;
    percentage?: number;
  };
  processing_time_ms?: number;
  error_message?: string;
}

export async function uploadImage(file: File): Promise<AnalysisStatus> {
  const form = new FormData();
  form.append('image', file);
  const { data } = await api.post('/api/v1/analyze', form);
  return data;
}

export async function getAnalysisStatus(id: string): Promise<AnalysisStatus> {
  const { data } = await api.get(`/api/v1/analysis/${id}`);
  return data;
}

export async function getAnalysisReport(id: string): Promise<any> {
  const { data } = await api.get(`/api/v1/analysis/${id}/report`);
  return data;
}

export function getHeatmapUrl(id: string): string {
  return `${API_BASE}/api/v1/analysis/${id}/heatmap`;
}

export function getAnalysisImageUrl(id: string): string {
  return `${API_BASE}/api/v1/analysis/${encodeURIComponent(id)}/image`;
}

export async function getHealth(): Promise<any> {
  const { data } = await api.get('/api/v1/health');
  return data;
}

export async function getModels(): Promise<any> {
  const { data } = await api.get('/api/v1/models');
  return data;
}
