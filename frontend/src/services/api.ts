export type JobStatusPayload = {
  status: string;
  progress?: number;
  detail?: string;
  finished?: boolean;
  exit_code?: number | null;
};

export type MetricsPayload = {
  step?: number;
  loss?: number;
  diffusion_loss?: number;
  lr?: number;
  throughput?: number;
  vram_gb?: number;
  preprocess_percent?: number;
  preprocess_count?: number;
  preprocess_total?: number;
  preprocess_speed?: number;
  preprocess_elapsed?: number;
  preprocess_remaining?: number;
  cpu_percent?: number;
};

export type SystemPayload = {
  cpu_percent?: number | null;
};

async function postJson<T>(url: string, payload: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json() as Promise<T>;
}

export async function importZip(params: {
  zipFile?: File | null;
  outDir: string;
  clearDir: boolean;
}): Promise<{ message: string; out_dir: string }>{
  const fd = new FormData();
  fd.append("out_dir", params.outDir);
  fd.append("clear_dir", String(params.clearDir));
  if (params.zipFile) {
    fd.append("zip_file", params.zipFile);
  }
  const res = await fetch("/api/import", { method: "POST", body: fd });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

export async function startPreprocess(payload: Record<string, unknown>): Promise<{ job_id: string }>{
  return postJson("/api/preprocess", payload);
}

export async function startTrain(payload: Record<string, unknown>): Promise<{ job_id: string; metrics_path?: string }>{
  return postJson("/api/train", payload);
}

export async function startAnalysis(payload: Record<string, unknown>): Promise<{ job_id: string }>{
  return postJson("/api/analysis", payload);
}

export async function startGenerate(payload: Record<string, unknown>): Promise<{ job_id: string }>{
  return postJson("/api/generate", payload);
}

export async function startRenderFlac(payload: Record<string, unknown>): Promise<{ job_id: string }>{
  return postJson("/api/render_flac", payload);
}

export function streamLogs(
  jobId: string,
  onLog: (line: string) => void,
  onStatus: (payload: JobStatusPayload) => void,
  onError?: (err: Event) => void
): () => void {
  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  es.addEventListener("log", (ev) => {
    try {
      const payload = JSON.parse((ev as MessageEvent).data);
      if (payload?.line) onLog(payload.line);
    } catch {
      // ignore
    }
  });
  es.addEventListener("status", (ev) => {
    try {
      const payload = JSON.parse((ev as MessageEvent).data);
      onStatus(payload);
    } catch {
      // ignore
    }
  });
  es.onerror = (err) => {
    if (onError) onError(err as Event);
  };
  return () => es.close();
}

export function streamMetrics(
  jobId: string,
  onMetric: (payload: MetricsPayload) => void,
  onError?: (err: Event) => void
): () => void {
  const es = new EventSource(`/api/jobs/${jobId}/metrics`);
  es.addEventListener("metrics", (ev) => {
    try {
      const payload = JSON.parse((ev as MessageEvent).data);
      onMetric(payload);
    } catch {
      // ignore
    }
  });
  es.onerror = (err) => {
    if (onError) onError(err as Event);
  };
  return () => es.close();
}

export function streamSystem(
  onSystem: (payload: SystemPayload) => void,
  onError?: (err: Event) => void
): () => void {
  const es = new EventSource("/api/system/stream");
  es.addEventListener("system", (ev) => {
    try {
      const payload = JSON.parse((ev as MessageEvent).data);
      onSystem(payload);
    } catch {
      // ignore
    }
  });
  es.onerror = (err) => {
    if (onError) onError(err as Event);
  };
  return () => es.close();
}
