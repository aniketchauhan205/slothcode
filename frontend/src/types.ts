export type JobStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

export interface Job {
  id: string;
  prompt: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  error: string | null;
  plan: Record<string, unknown> | null;
  preview_url: string | null;
  file_count: number;
}

export interface JobEvent {
  type: string;
  data: Record<string, unknown>;
  timestamp: string;
}

export interface PreviewStatus {
  running: boolean;
  preview_url: string | null;
  port?: number;
}

export interface FileContent {
  path: string;
  content: string;
}
