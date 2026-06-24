import { Handle, Position, type NodeProps } from "@xyflow/react";

export default function DynamicNode({ data, selected }: NodeProps) {
  const d = data as Record<string, unknown>;
  return (
    <div
      style={{
        background: selected ? "var(--red-dim)" : "var(--bg-card)",
        border: `1.5px solid ${selected ? "var(--red)" : "var(--border)"}`,
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 140,
        cursor: "pointer",
        transition: "border-color 0.1s, background 0.1s",
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: "var(--red)" }} />
      <div style={{ color: "var(--red)", fontSize: 10, letterSpacing: "0.08em", marginBottom: 3 }}>
        DYNAMIC
      </div>
      <div style={{ color: "var(--text)", fontSize: 13, fontWeight: 500 }}>
        {String(d.label ?? d.impl ?? "dynamic")}
      </div>
      <Handle type="source" position={Position.Right} style={{ background: "var(--red)" }} />
    </div>
  );
}
