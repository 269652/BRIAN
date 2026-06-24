import { create } from "zustand";
import type { ArchSummary, ArchDetail, MechanicSpec, StudioNode, StudioEdge } from "./types";
import type { Node, Edge } from "@xyflow/react";
import { api } from "./api";

interface StudioState {
  // Architecture list
  archs: ArchSummary[];
  archsLoading: boolean;
  loadArchs: () => Promise<void>;

  // Active architecture
  activeArch: string | null;
  archDetail: ArchDetail | null;
  archLoading: boolean;
  openArch: (name: string) => Promise<void>;

  // React Flow state
  nodes: Node[];
  edges: Edge[];
  setNodes: (nodes: Node[]) => void;
  setEdges: (edges: Edge[]) => void;

  // Mechanic library
  mechanics: MechanicSpec[];
  mechanicsLoading: boolean;
  loadMechanics: () => Promise<void>;

  // Selection
  selectedNode: Node | null;
  setSelectedNode: (node: Node | null) => void;

  // Right panel tab
  rightPanel: "properties" | "inference";
  setRightPanel: (tab: "properties" | "inference") => void;

  // Status messages
  statusMsg: string;
  setStatus: (msg: string) => void;

  // Source editor
  sourceVisible: boolean;
  toggleSource: () => void;

  // Save
  saveArch: () => Promise<void>;

  // Deploy
  deploying: boolean;
  deployArch: (steps?: number) => Promise<void>;
}

function studioToFlow(nodes: StudioNode[], edges: StudioEdge[]): { nodes: Node[]; edges: Edge[] } {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type,
      position: n.position,
      data: n.data,
      style: {},
    })),
    edges: edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      animated: e.animated ?? false,
    })),
  };
}

export const useStore = create<StudioState>((set, get) => ({
  archs: [],
  archsLoading: false,
  async loadArchs() {
    set({ archsLoading: true });
    try {
      const archs = await api.architectures.list();
      set({ archs, archsLoading: false });
    } catch {
      set({ archsLoading: false });
    }
  },

  activeArch: null,
  archDetail: null,
  archLoading: false,
  async openArch(name) {
    set({ archLoading: true, activeArch: name });
    try {
      const detail = await api.architectures.get(name);
      const { nodes, edges } = studioToFlow(detail.nodes, detail.edges);
      set({ archDetail: detail, nodes, edges, archLoading: false, selectedNode: null });
    } catch (e) {
      set({ archLoading: false, statusMsg: `Failed to load ${name}` });
    }
  },

  nodes: [],
  edges: [],
  setNodes: (nodes) => set({ nodes }),
  setEdges: (edges) => set({ edges }),

  mechanics: [],
  mechanicsLoading: false,
  async loadMechanics() {
    set({ mechanicsLoading: true });
    try {
      const mechanics = await api.mechanics.list();
      set({ mechanics, mechanicsLoading: false });
    } catch {
      set({ mechanicsLoading: false });
    }
  },

  selectedNode: null,
  setSelectedNode: (node) => set({ selectedNode: node, rightPanel: node ? "properties" : get().rightPanel }),

  rightPanel: "properties",
  setRightPanel: (tab) => set({ rightPanel: tab }),

  statusMsg: "",
  setStatus: (msg) => set({ statusMsg: msg }),

  sourceVisible: false,
  toggleSource: () => set((s) => ({ sourceVisible: !s.sourceVisible })),

  async saveArch() {
    const { activeArch, archDetail } = get();
    if (!activeArch || !archDetail) return;
    set({ statusMsg: "Saving…" });
    try {
      await api.architectures.save(activeArch, archDetail.source);
      set({ statusMsg: "Saved." });
    } catch {
      set({ statusMsg: "Save failed." });
    }
  },

  deploying: false,
  async deployArch(steps = 10000) {
    const { activeArch } = get();
    if (!activeArch) return;
    set({ deploying: true, statusMsg: "Deploying to vast.ai…" });
    try {
      const r = await api.deploy.start(activeArch, steps);
      set({
        deploying: false,
        statusMsg: r.success ? "Deploy started." : `Deploy failed: ${r.stderr.slice(0, 120)}`,
      });
    } catch (e) {
      set({ deploying: false, statusMsg: "Deploy error." });
    }
  },
}));
