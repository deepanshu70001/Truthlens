import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '';

const api = axios.create({ baseURL: API_BASE, timeout: 120_000 });
const TOKEN_KEY = 'truthlens.accessToken';

export function getAccessToken(): string | null {
  return typeof window === 'undefined' ? null : window.sessionStorage.getItem(TOKEN_KEY);
}

export function setAccessToken(token: string | null): void {
  if (typeof window === 'undefined') return;
  if (token) window.sessionStorage.setItem(TOKEN_KEY, token);
  else window.sessionStorage.removeItem(TOKEN_KEY);
}

api.interceptors.request.use(config => {
  const token = getAccessToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(response => response, error => {
  if (error.response?.status === 401 && typeof window !== 'undefined') {
    setAccessToken(null);
    window.dispatchEvent(new CustomEvent('truthlens:auth-expired'));
  }
  return Promise.reject(error);
});

export interface AnalysisStatus {
  analysis_id: string;
  job?: { job_id: string; state: string; attempt_count: number; max_attempts: number; error_category?: string; error_message?: string };
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

export interface AnalysisReport {
  analysis_id: string;
  status: string;
  stages?: Record<string, unknown>;
  [key: string]: unknown;
}

export async function getAuthMode(): Promise<{ auth_required: boolean }> {
  const { data } = await api.get('/api/v1/auth/mode');
  return data;
}

export async function login(username: string, password: string): Promise<void> {
  const { data } = await api.post('/api/v1/auth/token', { username, password });
  setAccessToken(data.access_token);
}

export function logout(): void {
  setAccessToken(null);
}

export async function uploadImage(file: File, idempotencyKey: string): Promise<AnalysisStatus> {
  const form = new FormData();
  form.append('image', file);
  const { data } = await api.post('/api/v1/analyze', form, { headers: { 'Idempotency-Key': idempotencyKey } });
  return data;
}

export async function getAnalysisStatus(id: string): Promise<AnalysisStatus> {
  const { data } = await api.get(`/api/v1/analysis/${id}`);
  return data;
}

export async function getAnalysisReport(id: string): Promise<AnalysisReport> {
  const { data } = await api.get(`/api/v1/analysis/${id}/report`);
  return data;
}

export function getHeatmapUrl(id: string): string {
  return `${API_BASE}/api/v1/analysis/${id}/heatmap`;
}

export function getAnalysisImageUrl(id: string): string {
  return `${API_BASE}/api/v1/analysis/${encodeURIComponent(id)}/image`;
}

export function normalizeProtectedAssetUrl(url: string): string {
  const browserOrigin = typeof window === 'undefined' ? 'http://localhost' : window.location.origin;
  const apiOrigin = new URL(API_BASE || browserOrigin, browserOrigin).origin;
  const target = new URL(url, API_BASE || browserOrigin);
  const allowedPath = /^\/api\/v1\/analysis\/[0-9a-f-]{36}(?:\/(?:image|heatmap|forensics\/ela))?$/i;
  if (target.origin !== apiOrigin || !allowedPath.test(target.pathname)) {
    throw new Error('Protected resource URL is not a TruthLens analysis artifact.');
  }
  return `${target.origin}${target.pathname}${target.search}`;
}

export async function getProtectedAsset(url: string): Promise<Blob> {
  const { data } = await api.get(normalizeProtectedAssetUrl(url), { responseType: 'blob' });
  return data;
}

export async function getHealth(): Promise<any> {
  const { data } = await api.get('/api/v1/health');
  return data;
}

export async function getModels(): Promise<any> {
  const { data } = await api.get('/api/v1/models');
  return data;
}
