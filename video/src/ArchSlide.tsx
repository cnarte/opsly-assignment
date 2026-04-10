import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

const SERVICES = [
  { name: "Streamlit UI", color: "#3B82F6", icon: "💬", desc: "Chat + Graph View" },
  { name: "Gateway", color: "#8B5CF6", icon: "🔀", desc: "FastAPI REST + WebSocket" },
  { name: "Orchestrator", color: "#EC4899", icon: "🎭", desc: "LangGraph Supervisor" },
  { name: "Graph Query", color: "#10B981", icon: "🕸️", desc: "Neo4j Cypher queries" },
  { name: "Code Analyst", color: "#F59E0B", icon: "🧠", desc: "LLM code analysis" },
  { name: "Memory", color: "#06B6D4", icon: "💾", desc: "Redis + Neo4j memory" },
  { name: "Indexer", color: "#EF4444", icon: "📥", desc: "AST parser + embedder" },
];

export const ArchSlide: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const opacity = interpolate(
    frame,
    [0, 12, durationInFrames - 12, durationInFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: "#FFFFFF",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        opacity,
        padding: "40px 80px",
        gap: 32,
      }}
    >
      <h2 style={{
        margin: 0,
        fontSize: 40,
        fontWeight: 700,
        color: "#0F172A",
        fontFamily: "system-ui, sans-serif",
        letterSpacing: "-0.02em",
      }}>
        System Architecture
      </h2>
      <p style={{
        margin: 0,
        fontSize: 18,
        color: "#64748B",
        fontFamily: "system-ui, sans-serif",
        fontSize: 18,
      }}>
        7 specialised microservices communicating over MCP (Model Context Protocol)
      </p>

      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(4, 1fr)",
        gap: 16,
        width: "100%",
        maxWidth: 1100,
      }}>
        {SERVICES.map((svc, i) => {
          const delay = i * 4;
          const scale = spring({
            frame: Math.max(0, frame - delay),
            fps,
            config: { damping: 50, stiffness: 80 },
            from: 0.7,
            to: 1,
          });
          const itemOpacity = interpolate(
            frame,
            [delay, delay + 10],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div
              key={svc.name}
              style={{
                background: "#F8FAFC",
                border: `1px solid ${svc.color}44`,
                borderRadius: 12,
                padding: "20px 24px",
                transform: `scale(${scale})`,
                opacity: itemOpacity,
              }}
            >
              <div style={{ fontSize: 28, marginBottom: 8 }}>{svc.icon}</div>
              <div style={{
                color: svc.color,
                fontSize: 16,
                fontWeight: 700,
                fontFamily: "system-ui, sans-serif",
                marginBottom: 4,
              }}>
                {svc.name}
              </div>
              <div style={{
                color: "#64748B",
                fontSize: 13,
                fontFamily: "system-ui, sans-serif",
              }}>
                {(svc as any).desc}
              </div>
            </div>
          );
        })}

        {/* Neo4j + Redis */}
        {[
          { name: "Neo4j", color: "#22C55E", icon: "🗄️", desc: "Knowledge graph DB" },
          { name: "Redis", color: "#EF4444", icon: "⚡", desc: "Session cache" },
        ].map((svc, i) => {
          const delay = (SERVICES.length + i) * 4;
          const scale = spring({
            frame: Math.max(0, frame - delay),
            fps,
            config: { damping: 50, stiffness: 80 },
            from: 0.7,
            to: 1,
          });
          const itemOpacity = interpolate(
            frame,
            [delay, delay + 10],
            [0, 1],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
          );
          return (
            <div
              key={svc.name}
              style={{
                background: "#F8FAFC",
                border: `1px solid ${svc.color}44`,
                borderRadius: 12,
                padding: "20px 24px",
                transform: `scale(${scale})`,
                opacity: itemOpacity,
              }}
            >
              <div style={{ fontSize: 28, marginBottom: 8 }}>{svc.icon}</div>
              <div style={{
                color: svc.color,
                fontSize: 16,
                fontWeight: 700,
                fontFamily: "system-ui, sans-serif",
                marginBottom: 4,
              }}>
                {svc.name}
              </div>
              <div style={{
                color: "#64748B",
                fontSize: 13,
                fontFamily: "system-ui, sans-serif",
              }}>
                {(svc as any).desc}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
