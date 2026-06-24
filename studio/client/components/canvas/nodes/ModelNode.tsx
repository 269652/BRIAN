import { Handle, Position, type NodeProps } from "@xyflow/react";

export default function ModelNode({ data, selected }: NodeProps) {
  const d = data as Record<string, unknown>;
  return (
    <div
      style={{
        background: selected ? "var(--blue-dim)" : "var(--bg-card)",
        border: `1.5px solid ${selected ? "var(--blue)" : "var(--border)"}`,
        borderRadius: 8,
        padding: "10px 14px",
        minWidth: 160,
        cursor: "pointer",
        transition: "border-color 0.1s, background 0.1s",
      }}
    >
      <div style={{ color: "var(--blue)", fontSize: 10, letterSpacing: "0.08em", marginBottom: 4 }}>
        MODEL
      </div>
      <div style={{ color: "var(--text)", fontWeight: 600, fontSize: 14, marginBottom: 6 }}>
        {String(d.kind ?? "custom")}
      </div>
      {Boolean(d.weights) && (
        <div style={{ color: "var(--text-muted)", fontSize: 11, wordBreak: "break-all" }}>
          {String(d.weights)}
        </div>
      )}
      <Handle type="source" position={Position.Right} style={{ background: "var(--blue)" }} />
    </div>
  );
}
