"use client";
import dynamic from "next/dynamic";

// Avoid SSR for React Flow (uses browser APIs)
const StudioApp = dynamic(() => import("@/components/StudioApp"), {
  ssr: false,
  loading: () => (
    <div
      style={{
        height: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--bg)",
        color: "var(--text-muted)",
        fontFamily: "monospace",
        fontSize: 13,
        letterSpacing: "0.05em",
      }}
    >
      brian studio · loading…
    </div>
  ),
});

export default function Page() {
  return <StudioApp />;
}
