"use client";

import { useStore } from "@/lib/store";
import { useState } from "react";

export default function StudioToolbar() {
  const {
    activeArch,
    archs,
    openArch,
    saveArch,
    deployArch,
    deploying,
    statusMsg,
    toggleSource,
    sourceVisible,
  } = useStore();

  const [deploySteps, setDeploySteps] = useState(10000);
  const [showDeployMenu, setShowDeployMenu] = useState(false);

  return (
    <div
      style={{
        height: 44,
        background: "var(--bg-panel)",
        borderBottom: "1px solid var(--border)",
        display: "flex",
        alignItems: "center",
        padding: "0 12px",
        gap: 8,
        flexShrink: 0,
        position: "relative",
      }}
    >
      {/* Logo */}
      <span
        style={{
          color: "var(--accent)",
          fontWeight: 700,
          fontSize: 13,
          letterSpacing: "0.05em",
          marginRight: 8,
        }}
      >
        ⬡ brian studio
      </span>

      {/* Architecture selector */}
      <select
        value={activeArch ?? ""}
        onChange={(e) => e.target.value && openArch(e.target.value)}
        style={{
          background: "var(--bg-card)",
          border: "1px solid var(--border)",
          borderRadius: 5,
          color: "var(--text)",
          padding: "4px 8px",
          fontSize: 12,
          cursor: "pointer",
          minWidth: 140,
        }}
      >
        <option value="">select arch…</option>
        {archs.map((a) => (
          <option key={a.name} value={a.name}>
            {a.name}
          </option>
        ))}
      </select>

      <div style={{ width: 1, height: 20, background: "var(--border)", margin: "0 4px" }} />

      {/* Save */}
      <Btn
        label="Save"
        onClick={saveArch}
        disabled={!activeArch}
        color="var(--text-muted)"
      />

      {/* Compile */}
      <Btn
        label="Compile"
        onClick={async () => {
          if (!activeArch) return;
          const { compile } = (await import("@/lib/api")).api.architectures;
          const r = await compile(activeArch);
          useStore.getState().setStatus(r.success ? "Compiled." : r.stderr.slice(0, 80));
        }}
        disabled={!activeArch}
        color="var(--cyan)"
      />

      {/* Source toggle */}
      <Btn
        label={sourceVisible ? "Canvas" : "Source"}
        onClick={toggleSource}
        disabled={!activeArch}
        color="var(--text-muted)"
      />

      <div style={{ flex: 1 }} />

      {/* Deploy menu */}
      <div style={{ position: "relative" }}>
        <Btn
          label={deploying ? "Deploying…" : "Deploy →"}
          onClick={() => setShowDeployMenu((v) => !v)}
          disabled={!activeArch || deploying}
          color="var(--orange)"
        />
        {showDeployMenu && (
          <div
            style={{
              position: "absolute",
              top: "calc(100% + 6px)",
              right: 0,
              background: "var(--bg-card)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: 12,
              width: 200,
              zIndex: 100,
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            <label style={{ color: "var(--text-muted)", fontSize: 11 }}>
              Steps
              <input
                type="number"
                value={deploySteps}
                onChange={(e) => setDeploySteps(Number(e.target.value))}
                step={1000}
                min={100}
                style={{
                  background: "var(--bg)",
                  border: "1px solid var(--border)",
                  borderRadius: 4,
                  color: "var(--text)",
                  padding: "3px 6px",
                  fontSize: 12,
                  width: "100%",
                  marginTop: 4,
                }}
              />
            </label>
            <button
              onClick={() => {
                setShowDeployMenu(false);
                deployArch(deploySteps);
              }}
              style={{
                background: "var(--orange-dim)",
                border: "1px solid var(--orange)",
                borderRadius: 5,
                color: "var(--orange)",
                padding: "5px 10px",
                fontSize: 12,
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Deploy to vast.ai
            </button>
          </div>
        )}
      </div>

      {/* Status */}
      {statusMsg && (
        <span style={{ color: "var(--text-muted)", fontSize: 11, marginLeft: 8 }}>
          {statusMsg}
        </span>
      )}
    </div>
  );
}

function Btn({
  label,
  onClick,
  disabled,
  color,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  color?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: "transparent",
        border: "1px solid var(--border)",
        borderRadius: 5,
        color: disabled ? "var(--text-dim)" : color ?? "var(--text)",
        padding: "4px 10px",
        fontSize: 12,
        cursor: disabled ? "default" : "pointer",
        fontFamily: "inherit",
        transition: "border-color 0.1s, color 0.1s",
      }}
    >
      {label}
    </button>
  );
}
