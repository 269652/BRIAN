import { Handle, Position, type NodeProps } from "@xyflow/react";

export default function SheafNode({ data, selected }: NodeProps) {
  const d = data as Record<string, unknown>;
  const badges: string[] = [
    d.dim ? `dim=${d.dim}` : "",
    d.depth ? `×${d.depth}` : "",
    d.heads ? `h=${d.heads}` : "",
    d.context ? `ctx=${d.context}` : "",
  ].filter(Boolean) as string[];

  return (
    <div
      style={{
        background: selected ? "var(--purple-dim)" : "var(--bg-card)",
        border: `1.5px solid ${selected ? "var(--purple)" : "var(--border)"}`,
        borderRadius: 8,
        padding: "10px 14px",
        minWidth: 180,
        cursor: "pointer",
        transition: "border-color 0.1s, background 0.1s",
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: "var(--purple)" }} />
      <div style={{ color: "var(--purple)", fontSize: 10, letterSpacing: "0.08em", marginBottom: 4 }}>
        SHEAF
      </div>
      <div style={{ color: "var(--text)", fontWeight: 600, fontSize: 13, marginBottom: 8 }}>
        {String(d.norm_name ?? (typeof d.norm === "string" ? d.norm : "transformer"))}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
        {badges.map((b, i) => (
          <span
            key={i}
            style={{
              background: "var(--purple-dim)",
              color: "var(--purple)",
              borderRadius: 4,
              padding: "2px 6px",
              fontSize: 11,
            }}
          >
            {b}
          </span>
        ))}
      </div>
      <Handle type="source" position={Position.Right} style={{ background: "var(--purple)" }} />
    </div>
  );
}
