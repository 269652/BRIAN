"use client";

import { useStore } from "@/lib/store";
import { useState } from "react";
import type { Node } from "@xyflow/react";

export default function PropertiesPanel() {
  const { selectedNode, archDetail } = useStore();

  if (!selectedNode) {
    return (
      <div
        style={{
          padding: "20px 14px",
          color: "var(--text-dim)",
          fontSize: 12,
          textAlign: "center",
        }}
      >
        Click a node to inspect it
      </div>
    );
  }

  return <NodeProperties node={selectedNode} />;
}

const SKIP_KEYS = new Set(["label"]);
const RICH_KEYS = new Set(["equation_math", "where", "primitive_properties"]);

function NodeProperties({ node }: { node: Node }) {
  const d = node.data as Record<string, unknown>;

  const typeColor: Record<string, string> = {
    model: "var(--blue)",
    sheaf: "var(--purple)",
    mechanic: "var(--accent)",
    structure: "var(--orange)",
    dynamic: "var(--red)",
    population: "var(--cyan)",
    neurotransmitter: "var(--orange)",
  };
  const color = typeColor[node.type ?? "mechanic"] ?? "var(--text-muted)";

  // Split keys: rich (equation/where) first, then plain props, skip label
  const richEntries = Object.entries(d).filter(([k]) => RICH_KEYS.has(k));
  const plainEntries = Object.entries(d).filter(([k]) => !SKIP_KEYS.has(k) && !RICH_KEYS.has(k));

  return (
    <div style={{ overflow: "auto", height: "100%" }}>
      <div
        style={{
          padding: "10px 14px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          gap: 8,
        }}
      >
        <span style={{ color, fontSize: 10, letterSpacing: "0.08em" }}>
          {(node.type ?? "node").toUpperCase()}
        </span>
        <span style={{ color: "var(--text)", fontSize: 13, fontWeight: 600 }}>
          {String(d.label ?? node.id)}
        </span>
      </div>

      {/* Rich: equation math */}
      {richEntries.map(([k, v]) => (
        <RichSection key={k} label={k} value={v} />
      ))}

      {/* Plain key/value pairs */}
      <div style={{ padding: "10px 14px", display: "flex", flexDirection: "column", gap: 2 }}>
        {plainEntries.map(([k, v]) => (
          <PropRow key={k} label={k} value={v} />
        ))}
      </div>
    </div>
  );
}

function RichSection({ label, value }: { label: string; value: unknown }) {
  const title = label === "equation_math" ? "equation"
    : label === "primitive_properties" ? "properties"
    : label;

  if (label === "equation_math" && typeof value === "string") {
    return (
      <div style={{ borderBottom: "1px solid var(--border)" }}>
        <div style={{ padding: "6px 14px 2px", color: "var(--text-muted)", fontSize: 10, letterSpacing: "0.06em" }}>
          {title.toUpperCase()}
        </div>
        <pre style={{
          margin: 0,
          padding: "6px 14px 10px",
          fontSize: 11,
          color: "var(--accent)",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
          lineHeight: 1.5,
          fontFamily: "inherit",
        }}>
          {value}
        </pre>
      </div>
    );
  }

  if (typeof value === "object" && value !== null) {
    const entries = Object.entries(value as Record<string, unknown>);
    return (
      <div style={{ borderBottom: "1px solid var(--border)" }}>
        <div style={{ padding: "6px 14px 2px", color: "var(--text-muted)", fontSize: 10, letterSpacing: "0.06em" }}>
          {title.toUpperCase()}
        </div>
        <div style={{ padding: "2px 14px 8px", display: "flex", flexDirection: "column", gap: 2 }}>
          {entries.map(([k, v]) => (
            <PropRow key={k} label={k} value={v} />
          ))}
        </div>
      </div>
    );
  }

  return <PropRow label={label} value={value} />;
}

function PropRow({ label, value }: { label: string; value: unknown }) {
  const [editing, setEditing] = useState(false);
  const [val, setVal] = useState(String(value ?? ""));

  const display = typeof value === "boolean"
    ? value ? "true" : "false"
    : String(value ?? "—");

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "100px 1fr",
        gap: 4,
        alignItems: "start",
        padding: "3px 0",
        borderBottom: "1px solid var(--border)",
      }}
    >
      <span style={{ color: "var(--text-muted)", fontSize: 11, paddingTop: 1 }}>{label}</span>
      {editing ? (
        <input
          autoFocus
          value={val}
          onChange={(e) => setVal(e.target.value)}
          onBlur={() => setEditing(false)}
          onKeyDown={(e) => e.key === "Enter" && setEditing(false)}
          style={{
            background: "var(--bg)",
            border: "1px solid var(--border-light)",
            borderRadius: 3,
            color: "var(--text)",
            padding: "1px 5px",
            fontSize: 12,
            fontFamily: "inherit",
            outline: "none",
          }}
        />
      ) : (
        <span
          onClick={() => setEditing(true)}
          style={{
            color: "var(--text)",
            fontSize: 12,
            cursor: "text",
            wordBreak: "break-all",
          }}
          title="Click to edit"
        >
          {display}
        </span>
      )}
    </div>
  );
}
