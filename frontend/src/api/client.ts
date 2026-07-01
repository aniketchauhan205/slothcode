import type { FileContent, Job, JobEvent, PreviewStatus } from "../types";

const API = import.meta.env.VITE_API_URL || "/api";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${url}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export async function createJob(prompt: string): Promise<Job> {
  return request<Job>("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
}

export async function getJob(jobId: string): Promise<Job> {
  return request<Job>(`/jobs/${jobId}`);
}

export async function cancelJob(jobId: string): Promise<Job> {
  return request<Job>(`/jobs/${jobId}/cancel`, { method: "POST" });
}

export async function listFiles(jobId: string): Promise<string[]> {
  const data = await request<{ files: string[] }>(`/jobs/${jobId}/files`);
  return data.files;
}

export async function getFileContent(jobId: string, path: string): Promise<FileContent> {
  return request<FileContent>(`/jobs/${jobId}/files/content?path=${encodeURIComponent(path)}`);
}

export function getDownloadUrl(jobId: string): string {
  return `${API}/jobs/${jobId}/download`;
}

export async function startPreview(
  jobId: string,
  files?: Record<string, string>,
): Promise<{ preview_url: string; message?: string }> {
  return request(`/jobs/${jobId}/preview/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files: files ?? null }),
  });
}

export async function getPreviewStatus(jobId: string): Promise<PreviewStatus> {
  return request<PreviewStatus>(`/jobs/${jobId}/preview`);
}

export async function stopPreview(jobId: string): Promise<void> {
  await request(`/jobs/${jobId}/preview/stop`, { method: "POST" });
}

export function subscribeToJobEvents(
  jobId: string,
  onEvent: (event: JobEvent) => void,
  onError?: (err: Event) => void,
): () => void {
  const source = new EventSource(`${API}/jobs/${jobId}/events`);

  source.onmessage = (msg) => {
    try {
      const event = JSON.parse(msg.data) as JobEvent;
      onEvent(event);
    } catch {
      // ignore malformed events
    }
  };

  source.onerror = (err) => {
    onError?.(err);
  };

  return () => source.close();
}

