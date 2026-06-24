import { Handle, Position, type NodeProps } from "@xyflow/react";

const CAT_COLORS: Record<string, string> = {
  attention: "var(--accent)",
  position: "var(--cyan)",
  ffn: "var(--orange)",
  norm: "var(--blue)",
  moe: "var(--red)",
  ssm: "var(--purple)",
  regularizer: "var(--text-muted)",
  routing: "var(--orange)",
};

export default function MechanicNode({ data, selected }: NodeProps) {
  const d = data as Record<string, unknown>;
  const cat = String(d.category ?? "mechanic");
  const color = CAT_COLORS[cat] ?? "var(--text-muted)";

  return (
    <div
      style={{
        background: selected ? "var(--bg-hover)" : "var(--bg-card)",
        border: `1.5px solid ${selected ? color : "var(--border)"}`,
        borderRadius: 8,
        padding: "8px 12px",
        minWidth: 150,
        cursor: "pointer",
        transition: "border-color 0.1s, background 0.1s",
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: color }} />
      <div style={{ color, fontSize: 10, letterSpacing: "0.08em", marginBottom: 3 }}>
        {cat.toUpperCase()}
      </div>
      <div style={{ color: "var(--text)", fontSize: 13, fontWeight: 500 }}>
        {String(d.label ?? d.impl ?? d.name ?? "mechanic")}
      </div>
      <Handle type="source" position={Position.Right} style={{ background: color }} />
    </div>
  );
}
