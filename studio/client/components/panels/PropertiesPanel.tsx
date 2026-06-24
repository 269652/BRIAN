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

function NodeProperties({ node }: { node: Node }) {
  const d = node.data as Record<string, unknown>;

  const typeColor: Record<string, string> = {
    model: "var(--blue)",
    sheaf: "var(--purple)",
    mechanic: "var(--accent)",
    structure: "var(--orange)",
    dynamic: "var(--red)",
  };
  const color = typeColor[node.type ?? "mechanic"] ?? "var(--text-muted)";

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
          {node.id}
        </span>
      </div>

      <div style={{ padding: "10px 14px", display: "flex", flexDirection: "column", gap: 2 }}>
        {Object.entries(d).map(([k, v]) => {
          if (k === "label") return null;
          return (
            <PropRow key={k} label={k} value={v} />
          );
        })}
      </div>
    </div>
  );
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
