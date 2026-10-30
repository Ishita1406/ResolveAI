// The one place the frontend talks to the backend. Same origin: Vite (dev) and nginx
// (Docker) forward these paths to the API.

import type {
  ConversationReplay,
  ConversationSummary,
  EvalResults,
  EvalRunListing,
  VoiceAnswer,
  VoiceOffer,
} from "./types";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

// The admin token (only needed beyond localhost, when the backend sets ADMIN_TOKEN). Kept for
// this browser tab only; storage can be unavailable (private windows), which is fine.
const TOKEN_KEY = "resolveai.adminToken";

export function getAdminToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return "";
  }
}

export function setAdminToken(token: string): void {
  try {
    if (token) sessionStorage.setItem(TOKEN_KEY, token);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    // not persisted; the next request just won't carry it
  }
}

async function request<T>(path: string, init: RequestInit = {}, admin = false): Promise<T> {
  const headers = new Headers(init.headers);
  const token = admin ? getAdminToken() : "";
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let response: Response;
  try {
    response = await fetch(path, { ...init, headers });
  } catch {
    throw new ApiError(0, "The server can't be reached. Is the backend running?");
  }
  if (!response.ok) {
    let detail = response.statusText || `HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  /** WebRTC signalling: our offer (with every ICE candidate) in, the server's answer out. */
  voiceOffer: (offer: VoiceOffer) =>
    request<VoiceAnswer>("/voice/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(offer),
    }),

  conversations: (limit = 50) =>
    request<ConversationSummary[]>(`/admin/conversations?limit=${limit}`, {}, true),

  conversation: (id: number) =>
    request<ConversationReplay>(`/admin/conversations/${encodeURIComponent(id)}`, {}, true),

  evalRuns: () => request<EvalRunListing[]>("/admin/evals", {}, true),

  latestEval: () => request<EvalResults>("/admin/evals/latest", {}, true),

  evalRun: (name: string) =>
    request<EvalResults>(`/admin/evals/${encodeURIComponent(name)}`, {}, true),
};
