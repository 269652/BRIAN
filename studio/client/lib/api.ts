import type {
  ArchSummary,
  ArchDetail,
  MechanicSpec,
  InferResult,
  DeployResult,
} from "./types";

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${path} → ${r.status}`);
  return r.json();
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`PUT ${path} → ${r.status}`);
  return r.json();
}

export const api = {
  architectures: {
    list: () => get<ArchSummary[]>("/architectures"),
    get: (name: string) => get<ArchDetail>(`/architectures/${name}`),
    save: (name: string, source: string) =>
      put<{ saved: boolean }>(`/architectures/${name}`, { source }),
    compile: (name: string) =>
      post<{ success: boolean; stdout: string; stderr: string }>(
        `/architectures/${name}/compile`,
        {}
      ),
  },
  mechanics: {
    list: () => get<MechanicSpec[]>("/mechanics"),
    get: (name: string) => get<MechanicSpec>(`/mechanics/${name}`),
  },
  deploy: {
    start: (arch: string, steps = 10000, label = "") =>
      post<DeployResult>("/deploy", { arch, steps, label }),
    status: () => get<{ success: boolean; output: string }>("/deploy/status"),
  },
  inference: {
    run: (
      prompt: string,
      arch: string,
      max_new_tokens = 128,
      temperature = 0.8,
      checkpoint = ""
    ) =>
      post<InferResult>("/inference", {
        prompt,
        arch,
        max_new_tokens,
        temperature,
        checkpoint,
      }),
  },
};
