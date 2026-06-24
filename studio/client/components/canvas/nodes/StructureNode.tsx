import { Handle, Position, type NodeProps } from "@xyflow/react";

export default function StructureNode({ data, selected }: NodeProps) {
  const d = data as Record<string, unknown>;
  return (
    <div
      style={{
        background: selected ? "var(--orange-dim)" : "var(--bg-card)",
        border: `1.5px dashed ${selected ? "var(--orange)" : "var(--border-light)"}`,
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 140,
        cursor: "pointer",
        transition: "border-color 0.1s, background 0.1s",
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: "var(--orange)" }} />
      <div style={{ color: "var(--orange)", fontSize: 10, letterSpacing: "0.08em", marginBottom: 3 }}>
        STRUCTURE
      </div>
      <div style={{ color: "var(--text)", fontSize: 13, fontWeight: 500 }}>
        {String(d.label ?? d.impl ?? "structure")}
      </div>
      <Handle type="source" position={Position.Right} style={{ background: "var(--orange)" }} />
    </div>
  );
}
