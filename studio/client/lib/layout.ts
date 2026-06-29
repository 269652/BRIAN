import type { Node, Edge } from "@xyflow/react";

export type LayoutMode = "lr" | "grid" | "radial" | "spring";

const GRID_COLS = 5;
const GRID_COL_W = 220;
const GRID_ROW_H = 110;

function gridLayout(nodes: Node[]): Node[] {
  return nodes.map((n, i) => ({
    ...n,
    position: {
      x: (i % GRID_COLS) * GRID_COL_W + 40,
      y: Math.floor(i / GRID_COLS) * GRID_ROW_H + 40,
    },
  }));
}

function radialLayout(nodes: Node[]): Node[] {
  const n = nodes.length;
  const R = Math.max(180, n * 28);
  const cx = R + 100, cy = R + 100;
  return nodes.map((node, i) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    return {
      ...node,
      position: { x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle) },
    };
  });
}

function springLayout(nodes: Node[], edges: Edge[], iterations = 120): Node[] {
  if (nodes.length === 0) return nodes;
  const W = 1400, H = 900;
  const k = Math.sqrt((W * H) / nodes.length);

  // Seed positions from server (they're already somewhat sensible)
  const pos = nodes.map((n) => ({ x: n.position.x, y: n.position.y }));
  const idxMap: Record<string, number> = {};
  nodes.forEach((n, i) => { idxMap[n.id] = i; });

  for (let iter = 0; iter < iterations; iter++) {
    const disp = pos.map(() => ({ x: 0, y: 0 }));
    const temp = (W / 8) * Math.pow(1 - iter / iterations, 1.5);

    // Repulsion
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = pos[i].x - pos[j].x || 0.1;
        const dy = pos[i].y - pos[j].y || 0.1;
        const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
        const f = (k * k) / dist;
        disp[i].x += (dx / dist) * f;
        disp[i].y += (dy / dist) * f;
        disp[j].x -= (dx / dist) * f;
        disp[j].y -= (dy / dist) * f;
      }
    }

    // Attraction
    for (const e of edges) {
      const u = idxMap[e.source], v = idxMap[e.target];
      if (u === undefined || v === undefined) continue;
      const dx = pos[u].x - pos[v].x;
      const dy = pos[u].y - pos[v].y;
      const dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const f = (dist * dist) / k;
      disp[u].x -= (dx / dist) * f;
      disp[u].y -= (dy / dist) * f;
      disp[v].x += (dx / dist) * f;
      disp[v].y += (dy / dist) * f;
    }

    // Apply with temperature
    for (let i = 0; i < nodes.length; i++) {
      const dx = disp[i].x, dy = disp[i].y;
      const len = Math.max(0.001, Math.sqrt(dx * dx + dy * dy));
      pos[i].x = Math.max(0, Math.min(W, pos[i].x + (dx / len) * Math.min(len, temp)));
      pos[i].y = Math.max(0, Math.min(H, pos[i].y + (dy / len) * Math.min(len, temp)));
    }
  }

  return nodes.map((n, i) => ({ ...n, position: pos[i] }));
}

export function applyLayout(nodes: Node[], edges: Edge[], mode: LayoutMode): Node[] {
  // "lr" (hierarchical) = the server's own topological positions, untouched.
  if (mode === "lr") return nodes;

  // Group-aware: only top-level nodes (no parentId) are repositioned. Children
  // keep their positions relative to their parent panel, so they ride along.
  const top = nodes.filter((n) => !n.parentId);
  const children = nodes.filter((n) => n.parentId);

  // Only edges between two top-level nodes inform a force layout.
  const topIds = new Set(top.map((n) => n.id));
  const topEdges = edges.filter((e) => topIds.has(e.source) && topIds.has(e.target));

  let laidOut: Node[];
  switch (mode) {
    case "grid":   laidOut = gridLayout(top); break;
    case "radial": laidOut = radialLayout(top); break;
    case "spring": laidOut = springLayout(top, topEdges); break;
    default:       laidOut = top;
  }
  // Preserve original array order (React Flow needs parents before children).
  const byId: Record<string, Node> = {};
  laidOut.forEach((n) => { byId[n.id] = n; });
  return nodes.map((n) => byId[n.id] ?? n);
}
