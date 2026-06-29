import { type NodeProps } from "@xyflow/react";

// Colored container panels that visually group related mechanics.
const GROUP_COLORS: Record<string, string> = {
  embed: "var(--cyan)",
  block: "var(--purple)",
  output: "var(--orange)",
  nt: "var(--orange)",
  populations: "var(--accent)",
  trunk: "var(--accent)",   // cortex (LM gradient)
  bio: "var(--red)",        // limbic / neuromodulatory nuclei (detached)
  other: "var(--text-muted)",
};

// Convert a CSS var to an rgba tint for subtle panel fills.
const TINT: Record<string, string> = {
  "var(--accent)": "rgba(80, 200, 120, 0.05)",
  "var(--red)": "rgba(239, 83, 80, 0.05)",
  "var(--orange)": "rgba(255, 167, 38, 0.05)",
  "var(--text-muted)": "rgba(255, 255, 255, 0.02)",
};

export default function GroupNode({ data }: NodeProps) {
  const d = data as Record<string, unknown>;
  const color = GROUP_COLORS[String(d.kind ?? "")] ?? "var(--border-light)";
  // Outer gradient-scope containers vs inner brain-region module panels.
  const isScope = d.tier === "scope";

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        border: isScope ? `1.5px solid ${color}` : `1.5px dashed ${color}`,
        borderRadius: isScope ? 16 : 10,
        background: isScope ? (TINT[color] ?? "rgba(255,255,255,0.02)") : "rgba(255,255,255,0.015)",
        position: "relative",
        boxSizing: "border-box",
        opacity: isScope ? 0.95 : 1,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: isScope ? 9 : 7,
          left: isScope ? 14 : 10,
          color,
          fontSize: isScope ? 11 : 9,
          fontWeight: 700,
          letterSpacing: isScope ? "0.12em" : "0.08em",
          textTransform: "uppercase",
          pointerEvents: "none",
          opacity: isScope ? 1 : 0.85,
        }}
      >
        {String(d.label ?? "")}
      </div>
    </div>
  );
}
