"use client";

import { useStore } from "@/lib/store";

const KIND_ICON: Record<string, string> = {
  gpt2: "G2",
  llama: "LL",
  qwen2: "Q2",
  custom: "◆",
};

export default function ArchitectureBrowser() {
  const { archs, activeArch, openArch, archsLoading } = useStore();

  return (
    <div style={{ display: "flex", flexDirection: "column" }}>
      <div
        style={{
          color: "var(--text-muted)",
          fontSize: 10,
          letterSpacing: "0.1em",
          padding: "10px 12px 6px",
        }}
      >
        ARCHITECTURES
      </div>

      {archsLoading && (
        <div style={{ color: "var(--text-dim)", fontSize: 11, padding: "4px 12px" }}>
          loading…
        </div>
      )}

      {archs.map((a) => (
        <button
          key={a.name}
          onClick={() => openArch(a.name)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "7px 12px",
            background: activeArch === a.name ? "var(--bg-hover)" : "transparent",
            border: "none",
            borderLeft: `2px solid ${activeArch === a.name ? "var(--accent)" : "transparent"}`,
            color: activeArch === a.name ? "var(--text)" : "var(--text-muted)",
            fontSize: 12,
            cursor: "pointer",
            textAlign: "left",
            fontFamily: "inherit",
            width: "100%",
            transition: "background 0.1s, color 0.1s",
          }}
        >
          <span
            style={{
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              padding: "1px 5px",
              fontSize: 10,
              color: "var(--text-muted)",
              flexShrink: 0,
            }}
          >
            {KIND_ICON[a.kind] ?? a.kind.slice(0, 2).toUpperCase()}
          </span>
          {a.name}
        </button>
      ))}
    </div>
  );
}
