"use client";

import { useStore } from "@/lib/store";
import { useState, useRef, useEffect } from "react";
import { api } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function InferencePanel() {
  const { activeArch } = useStore();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [maxTokens, setMaxTokens] = useState(128);
  const [temp, setTemp] = useState(0.8);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || !activeArch || loading) return;
    const prompt = input.trim();
    setInput("");
    setMessages((m) => [...m, { role: "user", content: prompt }]);
    setLoading(true);
    try {
      const r = await api.inference.run(prompt, activeArch, maxTokens, temp);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: r.success ? r.output : `Error: ${r.error}` },
      ]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: "Request failed." }]);
    }
    setLoading(false);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Settings strip */}
      <div
        style={{
          display: "flex",
          gap: 8,
          padding: "8px 12px",
          borderBottom: "1px solid var(--border)",
          alignItems: "center",
          flexShrink: 0,
        }}
      >
        <label style={{ color: "var(--text-muted)", fontSize: 11, display: "flex", gap: 4, alignItems: "center" }}>
          max
          <input
            type="number"
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            min={8}
            max={512}
            step={8}
            style={numStyle}
          />
        </label>
        <label style={{ color: "var(--text-muted)", fontSize: 11, display: "flex", gap: 4, alignItems: "center" }}>
          temp
          <input
            type="number"
            value={temp}
            onChange={(e) => setTemp(Number(e.target.value))}
            min={0}
            max={2}
            step={0.1}
            style={numStyle}
          />
        </label>
        {!activeArch && (
          <span style={{ color: "var(--text-dim)", fontSize: 11, marginLeft: "auto" }}>
            open an arch first
          </span>
        )}
      </div>

      {/* Messages */}
      <div style={{ flex: 1, overflowY: "auto", padding: "10px 12px" }}>
        {messages.length === 0 && (
          <div style={{ color: "var(--text-dim)", fontSize: 12, textAlign: "center", marginTop: 20 }}>
            {activeArch
              ? `Inference against ${activeArch} — type a prompt`
              : "Select an architecture first"}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ marginBottom: 12 }}>
            <div
              style={{
                color: m.role === "user" ? "var(--text-muted)" : "var(--accent)",
                fontSize: 10,
                letterSpacing: "0.08em",
                marginBottom: 3,
              }}
            >
              {m.role === "user" ? "YOU" : activeArch?.toUpperCase() ?? "MODEL"}
            </div>
            <div
              style={{
                color: "var(--text)",
                fontSize: 12,
                lineHeight: 1.6,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
              }}
            >
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div style={{ color: "var(--text-dim)", fontSize: 12, fontStyle: "italic" }}>
            generating…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div
        style={{
          display: "flex",
          gap: 6,
          padding: "8px 10px",
          borderTop: "1px solid var(--border)",
          flexShrink: 0,
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
          placeholder={activeArch ? "prompt…" : "no arch selected"}
          disabled={!activeArch || loading}
          style={{
            flex: 1,
            background: "var(--bg)",
            border: "1px solid var(--border)",
            borderRadius: 5,
            color: "var(--text)",
            padding: "5px 8px",
            fontSize: 12,
            fontFamily: "inherit",
            outline: "none",
          }}
        />
        <button
          onClick={send}
          disabled={!activeArch || loading || !input.trim()}
          style={{
            background: "var(--accent-dim)",
            border: "1px solid var(--accent)",
            borderRadius: 5,
            color: "var(--accent)",
            padding: "5px 10px",
            fontSize: 12,
            cursor: "pointer",
            fontFamily: "inherit",
          }}
        >
          Run
        </button>
      </div>
    </div>
  );
}

const numStyle: React.CSSProperties = {
  background: "var(--bg)",
  border: "1px solid var(--border)",
  borderRadius: 4,
  color: "var(--text)",
  padding: "2px 5px",
  fontSize: 11,
  width: 52,
  fontFamily: "inherit",
};
