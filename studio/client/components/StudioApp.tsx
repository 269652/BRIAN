"use client";

import { useEffect, useState } from "react";
import { useStore } from "@/lib/store";
import StudioToolbar from "./toolbar/StudioToolbar";
import ArchitectureBrowser from "./sidebar/ArchitectureBrowser";
import ComponentLibrary from "./sidebar/ComponentLibrary";
import StudioCanvas from "./canvas/StudioCanvas";
import PropertiesPanel from "./panels/PropertiesPanel";
import InferencePanel from "./panels/InferencePanel";

const LEFT_TABS = ["archs", "library"] as const;
type LeftTab = (typeof LEFT_TABS)[number];

const RIGHT_TABS = ["properties", "inference"] as const;
type RightTab = (typeof RIGHT_TABS)[number];

export default function StudioApp() {
  const { loadArchs, loadMechanics, activeArch, archDetail, sourceVisible } = useStore();
  const [leftTab, setLeftTab] = useState<LeftTab>("archs");
  const [rightTab, setRightTab] = useState<RightTab>("properties");

  useEffect(() => {
    loadArchs();
    loadMechanics();
  }, [loadArchs, loadMechanics]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh" }}>
      <StudioToolbar />

      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        {/* ── Left sidebar ─────────────────────────────────────── */}
        <div
          style={{
            width: 220,
            flexShrink: 0,
            background: "var(--bg-panel)",
            borderRight: "1px solid var(--border)",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          {/* Tab bar */}
          <div
            style={{
              display: "flex",
              borderBottom: "1px solid var(--border)",
              padding: "0 6px",
            }}
          >
            {LEFT_TABS.map((t) => (
              <button
                key={t}
                onClick={() => setLeftTab(t)}
                style={{
                  flex: 1,
                  padding: "8px 4px",
                  fontSize: 11,
                  background: "transparent",
                  border: "none",
                  borderBottom: `2px solid ${leftTab === t ? "var(--accent)" : "transparent"}`,
                  color: leftTab === t ? "var(--accent)" : "var(--text-dim)",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  transition: "color 0.1s",
                }}
              >
                {t}
              </button>
            ))}
          </div>

          <div style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
            {leftTab === "archs" ? <ArchitectureBrowser /> : <ComponentLibrary />}
          </div>
        </div>

        {/* ── Main canvas / source ─────────────────────────────── */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {sourceVisible && archDetail ? (
            <SourceEditor source={archDetail.source} archName={activeArch ?? ""} />
          ) : (
            <StudioCanvas />
          )}
        </div>

        {/* ── Right panel ──────────────────────────────────────── */}
        <div
          style={{
            width: 260,
            flexShrink: 0,
            background: "var(--bg-panel)",
            borderLeft: "1px solid var(--border)",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          {/* Tab bar */}
          <div
            style={{
              display: "flex",
              borderBottom: "1px solid var(--border)",
              padding: "0 6px",
            }}
          >
            {RIGHT_TABS.map((t) => (
              <button
                key={t}
                onClick={() => setRightTab(t)}
                style={{
                  flex: 1,
                  padding: "8px 4px",
                  fontSize: 11,
                  background: "transparent",
                  border: "none",
                  borderBottom: `2px solid ${
                    rightTab === t ? "var(--blue)" : "transparent"
                  }`,
                  color: rightTab === t ? "var(--blue)" : "var(--text-dim)",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  transition: "color 0.1s",
                }}
              >
                {t}
              </button>
            ))}
          </div>

          <div style={{ flex: 1, overflow: "hidden" }}>
            {rightTab === "properties" ? <PropertiesPanel /> : <InferencePanel />}
          </div>
        </div>
      </div>

      {/* Bottom status bar */}
      <StatusBar />
    </div>
  );
}

function StatusBar() {
  const { activeArch, nodes, edges } = useStore();
  return (
    <div
      style={{
        height: 22,
        background: "var(--bg-panel)",
        borderTop: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        padding: "0 12px",
        gap: 16,
        flexShrink: 0,
      }}
    >
      {activeArch && (
        <>
          <span style={{ color: "var(--text-muted)", fontSize: 10 }}>
            arch: <span style={{ color: "var(--text)" }}>{activeArch}</span>
          </span>
          <span style={{ color: "var(--text-muted)", fontSize: 10 }}>
            nodes: <span style={{ color: "var(--text)" }}>{nodes.length}</span>
          </span>
          <span style={{ color: "var(--text-muted)", fontSize: 10 }}>
            edges: <span style={{ color: "var(--text)" }}>{edges.length}</span>
          </span>
        </>
      )}
      <div style={{ flex: 1 }} />
      <span style={{ color: "var(--text-dim)", fontSize: 10 }}>
        server :1984 · studio :3141 · mcp /mcp
      </span>
    </div>
  );
}

function SourceEditor({ source, archName }: { source: string; archName: string }) {
  const { archDetail } = useStore();
  const [text, setText] = useState(source);

  useEffect(() => {
    setText(source);
  }, [source]);

  const save = async () => {
    if (!archName) return;
    const { save } = (await import("@/lib/api")).api.architectures;
    await save(archName, text);
    useStore.getState().setStatus("Source saved.");
  };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
      <div
        style={{
          padding: "6px 12px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "var(--bg-panel)",
        }}
      >
        <span style={{ color: "var(--text-muted)", fontSize: 11 }}>
          {archName}/arch.neuro
        </span>
        <button
          onClick={save}
          style={{
            background: "var(--accent-dim)",
            border: "1px solid var(--accent)",
            borderRadius: 4,
            color: "var(--accent)",
            padding: "3px 8px",
            fontSize: 11,
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          Save
        </button>
      </div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        style={{
          flex: 1,
          background: "var(--bg)",
          color: "var(--text)",
          border: "none",
          outline: "none",
          padding: "12px 14px",
          fontSize: 12,
          fontFamily: "inherit",
          lineHeight: 1.7,
          resize: "none",
          tabSize: 4,
        }}
      />
    </div>
  );
}
