import React from "react";
import { useCurrentFrame, useVideoConfig, interpolate, spring } from "remotion";

export const TitleCard: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  const opacity = interpolate(
    frame,
    [0, 18, durationInFrames - 12, durationInFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const titleY = spring({ frame, fps, config: { damping: 55, stiffness: 65 }, from: 36, to: 0 });
  const subY = spring({
    frame: Math.max(0, frame - 8),
    fps,
    config: { damping: 55, stiffness: 65 },
    from: 28,
    to: 0,
  });

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
        gap: 20,
      }}
    >
      {/* Top accent bar */}
      <div style={{
        width: 56,
        height: 5,
        borderRadius: 3,
        background: "#8B5CF6",
        transform: `translateY(${titleY}px)`,
      }} />

      <h1
        style={{
          margin: 0,
          fontSize: 64,
          fontWeight: 800,
          color: "#0F172A",
          fontFamily: "system-ui, sans-serif",
          letterSpacing: "-0.03em",
          transform: `translateY(${titleY}px)`,
        }}
      >
        Opsly Assignment Solution
      </h1>

      <p
        style={{
          margin: 0,
          fontSize: 22,
          color: "#64748B",
          fontFamily: "system-ui, sans-serif",
          fontWeight: 400,
          transform: `translateY(${subY}px)`,
        }}
      >
        Made by Ankit Kumar
      </p>
    </div>
  );
};
