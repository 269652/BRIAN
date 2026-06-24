"use client";

import { useStore } from "@/lib/store";
import { useState } from "react";
import type { MechanicSpec } from "@/lib/types";

const TABS = ["mechanic", "structure", "dynamic"] as const;
type Tab = (typeof TABS)[number];

const TAB_COLORS: Record<Tab, string> = {
  mechanic: "var(--accent)",
  structure: "var(--orange)",
  dynamic: "var(--red)",
};

export default function ComponentLibrary() {
  const { mechanics, mechanicsLoading } = useStore();
  const [tab, setTab] = useState<Tab>("mechanic");
  const [search, setSearch] = useState("");

  const filtered = mechanics.filter(
    (m) =>
      m.node_type === tab &&
      (search === "" ||
        m.name.includes(search) ||
        m.category.includes(search) ||
        m.summary.toLowerCase().includes(search.toLowerCase()))
  );

  const grouped: Record<string, MechanicSpec[]> = {};
  for (const m of filtered) {
    if (!grouped[m.category]) grouped[m.category] = [];
    grouped[m.category].push(m);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>
      <div
        style={{
          color: "var(--text-muted)",
          fontSize: 10,
          letterSpacing: "0.1em",
          padding: "10px 12px 6px",
        }}
      >
        COMPONENTS
      </div>

      {/* Tabs */}
      <div
        style={{
          display: "flex",
          borderBottom: "1px solid var(--border)",
          padding: "0 8px",
        }}
      >
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              flex: 1,
              padding: "5px 4px",
              fontSize: 11,
              background: "transparent",
              border: "none",
              borderBottom: `2px solid ${tab === t ? TAB_COLORS[t] : "transparent"}`,
              color: tab === t ? TAB_COLORS[t] : "var(--text-dim)",
              cursor: "pointer",
              fontFamily: "inherit",
              transition: "color 0.1s",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Search */}
      <input
        placeholder="search…"
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        style={{
          margin: "8px 10px 4px",
          background: "var(--bg)",
          border: "1px solid var(--border)",
          borderRadius: 5,
          color: "var(--text)",
          padding: "4px 8px",
          fontSize: 11,
          fontFamily: "inherit",
          outline: "none",
        }}
      />

      {/* List */}
      <div style={{ overflowY: "auto", flex: 1 }}>
        {mechanicsLoading && (
          <div style={{ color: "var(--text-dim)", fontSize: 11, padding: "8px 12px" }}>
            loading…
          </div>
        )}
        {Object.entries(grouped).map(([cat, items]) => (
          <div key={cat}>
            <div
              style={{
                color: "var(--text-dim)",
                fontSize: 10,
                letterSpacing: "0.08em",
                padding: "6px 12px 2px",
              }}
            >
              {cat.toUpperCase()}
            </div>
            {items.map((m) => (
              <ComponentItem key={m.name} mechanic={m} color={TAB_COLORS[tab]} />
            ))}
          </div>
        ))}
        {!mechanicsLoading && filtered.length === 0 && (
          <div style={{ color: "var(--text-dim)", fontSize: 11, padding: "8px 12px" }}>
            no results
          </div>
        )}
      </div>
    </div>
  );
}

function ComponentItem({ mechanic, color }: { mechanic: MechanicSpec; color: string }) {
  const [hover, setHover] = useState(false);

  const onDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData("application/brian-mechanic", JSON.stringify(mechanic));
    e.dataTransfer.effectAllowed = "copy";
  };

  return (
    <div
      draggable
      onDragStart={onDragStart}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      title={mechanic.summary}
      style={{
        padding: "5px 12px",
        cursor: "grab",
        background: hover ? "var(--bg-hover)" : "transparent",
        display: "flex",
        alignItems: "center",
        gap: 6,
        transition: "background 0.1s",
      }}
    >
      <span
        style={{
          width: 5,
          height: 5,
          borderRadius: "50%",
          background: color,
          flexShrink: 0,
        }}
      />
      <span style={{ color: "var(--text-muted)", fontSize: 12 }}>{mechanic.name}</span>
    </div>
  );
}
