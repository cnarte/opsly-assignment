import React from "react";
import {
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  Easing,
  Img,
  staticFile,
} from "remotion";

export type SlideProps = {
  imagePath: string;
  title: string;
  subtitle?: string;
  tag?: string;
  tagColor?: string;
  duration: number;
  zoomTarget?: { x: number; y: number; scale: number };
};

const FADE_FRAMES = 12;
const ZOOM_START = 10;

export const ScreenSlide: React.FC<SlideProps> = ({
  imagePath,
  title,
  subtitle,
  tag,
  tagColor = "#8B5CF6",
  duration,
  zoomTarget,
}) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // Fade in / fade out
  const opacity = interpolate(
    frame,
    [0, FADE_FRAMES, durationInFrames - FADE_FRAMES, durationInFrames],
    [0, 1, 1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Subtle zoom-in spring
  const zoomScale = spring({
    frame: Math.max(0, frame - ZOOM_START),
    fps,
    config: { damping: 60, stiffness: 80, mass: 1 },
    from: 1.04,
    to: zoomTarget?.scale ?? 1.0,
  });

  // Title slide-up
  const titleY = interpolate(frame, [0, FADE_FRAMES + 6], [24, 0], {
    easing: Easing.out(Easing.cubic),
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  const transformOrigin = zoomTarget
    ? `${zoomTarget.x}% ${zoomTarget.y}%`
    : "50% 50%";

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: "#F8FAFC",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        opacity,
      }}
    >
      {/* Screenshot */}
      <div
        style={{
          flex: 1,
          overflow: "hidden",
          position: "relative",
          borderRadius: "0 0 0 0",
        }}
      >
        <Img
          src={staticFile(imagePath)}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            objectPosition: "top left",
            transform: `scale(${zoomScale})`,
            transformOrigin,
          }}
        />
        {/* Subtle bottom vignette */}
        <div
          style={{
            position: "absolute",
            bottom: 0,
            left: 0,
            right: 0,
            height: 60,
            background: "linear-gradient(to top, rgba(248,250,252,0.6) 0%, transparent 100%)",
          }}
        />
      </div>

      {/* Bottom caption bar — light */}
      <div
        style={{
          padding: "24px 48px 28px",
          background: "#FFFFFF",
          borderTop: "2px solid #E2E8F0",
          transform: `translateY(${titleY}px)`,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14, marginBottom: 5 }}>
          {tag && (
            <span
              style={{
                background: tagColor,
                color: "#fff",
                borderRadius: 6,
                padding: "3px 12px",
                fontSize: 13,
                fontWeight: 700,
                letterSpacing: "0.06em",
                textTransform: "uppercase",
                fontFamily: "system-ui, sans-serif",
              }}
            >
              {tag}
            </span>
          )}
          <h2
            style={{
              margin: 0,
              fontSize: 26,
              fontWeight: 700,
              color: "#0F172A",
              fontFamily: "system-ui, sans-serif",
              letterSpacing: "-0.02em",
            }}
          >
            {title}
          </h2>
        </div>
        {subtitle && (
          <p
            style={{
              margin: 0,
              fontSize: 15,
              color: "#64748B",
              fontFamily: "system-ui, sans-serif",
              lineHeight: 1.5,
            }}
          >
            {subtitle}
          </p>
        )}
      </div>
    </div>
  );
};
