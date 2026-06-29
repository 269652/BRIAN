// Shared types for Brian Studio

export interface ArchSummary {
  name: string;
  has_config: boolean;
  has_fitness: boolean;
  kind: string;
}

export interface ArchDetail {
  name: string;
  source: string;
  nodes: StudioNode[];
  edges: StudioEdge[];
}

export interface StudioNode {
  id: string;
  type: "model" | "sheaf" | "mechanic" | "structure" | "dynamic" | "group" | string;
  data: Record<string, unknown>;
  position: { x: number; y: number };
  parentId?: string;
  extent?: string;
  style?: Record<string, unknown>;
}

export interface StudioEdge {
  id: string;
  source: string;
  target: string;
  animated?: boolean;
  label?: string;
  style?: Record<string, unknown>;
  data?: Record<string, unknown>;
}

export interface MechanicSpec {
  name: string;
  category: string;
  node_type: "mechanic" | "structure" | "dynamic";
  summary: string;
  equation: string;
  impl: string;
  zero_init: boolean;
  params: Record<string, ParamSpec>;
  when_to_use: string;
  not_for: string;
  properties: Record<string, string>;
  exported: boolean;
  file: string;
  dir: string;
}

export interface ParamSpec {
  default: unknown;
  type: string;
  min: number | null;
  max: number | null;
  doc: string;
}

export interface InferResult {
  success: boolean;
  output: string;
  error: string;
  prompt: string;
  arch: string;
}

export interface DeployResult {
  success: boolean;
  stdout: string;
  stderr: string;
}
