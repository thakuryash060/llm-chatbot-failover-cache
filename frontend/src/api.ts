export interface ResponseMetadata {
  model: string;
  resolution_source: "LIVE_INFERENCE" | "LLM_CACHE";
  key_id: string | null;
  fallback_attempts: number;
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
}

export interface ChatResponse {
  session_id: string;
  reply: string;
  metadata: ResponseMetadata;
}

export interface HealthResponse {
  status: string;
  keys_configured: number;
  cache_enabled: boolean;
}

const rawApiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const API_BASE_URL = rawApiUrl.replace(/\/+$/, "");

export function formatErrorDetail(detail: unknown, fallback: string): string {
  if (!detail) return fallback;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg);
        }
        return JSON.stringify(item);
      })
      .filter(Boolean);
    if (msgs.length > 0) return msgs.join("; ");
  }
  return fallback;
}

export async function sendChatMessage(params: {
  session_id: string;
  message: string;
  system_prompt?: string;
  simulate_failure?: string;
}): Promise<ChatResponse> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (params.simulate_failure && params.simulate_failure.trim()) {
    headers["X-Simulate-Failure"] = params.simulate_failure.trim();
  }

  const payload: Record<string, string> = {
    session_id: params.session_id,
    message: params.message,
  };
  if (params.system_prompt && params.system_prompt.trim()) {
    payload.system_prompt = params.system_prompt.trim();
  }

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
  } catch (err) {
    throw new Error(
      err instanceof Error
        ? `Network error: ${err.message}`
        : "Network failure: unable to reach backend server."
    );
  }

  let data: unknown = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    const detail =
      data && typeof data === "object" && "detail" in data
        ? (data as { detail: unknown }).detail
        : null;
    const message = formatErrorDetail(
      detail,
      `Request failed with HTTP ${response.status}`
    );
    throw new Error(message);
  }

  return data as ChatResponse;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed (${response.status})`);
  }
  return (await response.json()) as HealthResponse;
}
