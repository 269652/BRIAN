import { Handle, Position, type NodeProps } from "@xyflow/react";

const DYN_COLORS: Record<string, string> = {
  // Population dynamics types
  rate_code: "var(--accent)",
  oscillatory: "var(--purple)",
  integrate_and_fire: "var(--cyan)",
  gated: "var(--blue)",
  architecture: "var(--text-muted)",
  // NT node colors (dynamics = NT name)
  glutamate: "#00bcd4",
  gaba: "#ef5350",
  dopamine: "#ab47bc",
  serotonin: "#66bb6a",
  norepinephrine: "#ffa726",
  acetylcholine: "#42a5f5",
  endocannabinoid: "#ffee58",
};

export default function PopulationNode({ data, selected }: NodeProps) {
  const d = data as Record<string, unknown>;
  const dynamics = String(d.dynamics ?? d.category ?? "rate_code");
  // NT nodes carry an explicit per-transmitter colour; populations use dynamics.
  const color = (d.color as string) ?? DYN_COLORS[dynamics] ?? "var(--accent)";
  const count = d.count ? `n=${d.count}` : "";
  const tau = d.timescale ? `τ=${d.timescale}` : "";

  return (
    <div
      style={{
        background: selected ? "var(--bg-hover)" : "var(--bg-card)",
        border: `1.5px solid ${selected ? color : "var(--border)"}`,
        borderRadius: 6,
        padding: "6px 10px",
        minWidth: 120,
        maxWidth: 200,
        cursor: "pointer",
        transition: "border-color 0.1s, background 0.1s",
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: color }} />
      <div style={{ color, fontSize: 9, letterSpacing: "0.08em", marginBottom: 2 }}>
        {dynamics.toUpperCase().replace(/_/g, " ")}
      </div>
      <div style={{ color: "var(--text)", fontSize: 12, fontWeight: 600, marginBottom: 3 }}>
        {String(d.label ?? d.name ?? "?")}
      </div>
      {(count || tau) && (
        <div style={{ display: "flex", gap: 6 }}>
          {count && (
            <span style={{ color: "var(--text-muted)", fontSize: 10 }}>{count}</span>
          )}
          {tau && (
            <span style={{ color: "var(--text-muted)", fontSize: 10 }}>{tau}</span>
          )}
        </div>
      )}
      <Handle type="source" position={Position.Right} style={{ background: color }} />
    </div>
  );
}
