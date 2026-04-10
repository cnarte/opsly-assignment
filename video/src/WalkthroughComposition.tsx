import React from "react";
import { Sequence, useVideoConfig } from "remotion";
import { TitleCard } from "./TitleCard";
import { ArchSlide } from "./ArchSlide";
import { ScreenSlide } from "./ScreenSlide";
import { OutroCard } from "./OutroCard";

const FPS = 30;

// Slide durations in seconds → converted to frames
const sec = (s: number) => Math.round(s * FPS);

// Each slide: [component, durationSeconds]
// Total ~105s = ~3150 frames
export const SLIDES = [
  { type: "title",  durationSec: 6 },
  { type: "arch",   durationSec: 8 },
  {
    type: "screen",
    imagePath: "assets/screens/01_landing.png",
    title: "Clean Chat Interface",
    subtitle: "Ask anything about your codebase — sessions, memory, and repo scoping built-in.",
    tag: "UI",
    tagColor: "#3B82F6",
    durationSec: 6,
  },
  {
    type: "screen",
    imagePath: "assets/screens/02_query_submitted.png",
    title: "Sending a Query",
    subtitle: 'User asks "What classes inherit from APIRouter?" — the multi-agent pipeline kicks off.',
    tag: "Query",
    tagColor: "#8B5CF6",
    durationSec: 5,
  },
  {
    type: "screen",
    imagePath: "assets/screens/03_query_result.png",
    title: "Structured Answer",
    subtitle: "Graph query + synthesis LLM produce a concise, citation-backed response.",
    tag: "Answer",
    tagColor: "#10B981",
    durationSec: 8,
    zoomTarget: { x: 40, y: 60, scale: 1.06 },
  },
  {
    type: "screen",
    imagePath: "assets/screens/05_agent_activity.png",
    title: "Agent Activity Panel",
    subtitle: "See which agents ran — memory, graph_query — with live status badges.",
    tag: "Agents",
    tagColor: "#EC4899",
    durationSec: 6,
  },
  {
    type: "screen",
    imagePath: "assets/screens/04_graph_view.png",
    title: "Graph Visualization",
    subtitle: "Interactive Neo4j graph: APIRouter → INHERITS_FROM → Router, rendered in real-time.",
    tag: "Graph",
    tagColor: "#8B5CF6",
    durationSec: 8,
    zoomTarget: { x: 50, y: 45, scale: 1.08 },
  },
  {
    type: "screen",
    imagePath: "assets/screens/07_lifecycle_result.png",
    title: "Lifecycle Deep Dive",
    subtitle: 'Complex query: "Explain the lifecycle of a FastAPI request" — full ASGI flow explained.',
    tag: "Analysis",
    tagColor: "#F59E0B",
    durationSec: 9,
    zoomTarget: { x: 40, y: 55, scale: 1.05 },
  },
  {
    type: "screen",
    imagePath: "assets/screens/08_rich_graph.png",
    title: "Rich Symbol Context Graph",
    subtitle: "get_symbol_context returns 89 relationships — FastAPI at the center of its dependency web.",
    tag: "Graph",
    tagColor: "#8B5CF6",
    durationSec: 9,
    zoomTarget: { x: 50, y: 50, scale: 1.1 },
  },
  { type: "outro", durationSec: 8 },
];

export const TOTAL_FRAMES = SLIDES.reduce((acc, s) => acc + sec(s.durationSec), 0);

export const WalkthroughComposition: React.FC = () => {
  let offset = 0;

  return (
    <>
      {SLIDES.map((slide, i) => {
        const start = offset;
        const frames = sec(slide.durationSec);
        offset += frames;

        return (
          <Sequence key={i} from={start} durationInFrames={frames}>
            {slide.type === "title" && <TitleCard />}
            {slide.type === "arch" && <ArchSlide />}
            {slide.type === "outro" && <OutroCard />}
            {slide.type === "screen" && (
              <ScreenSlide
                imagePath={(slide as any).imagePath}
                title={(slide as any).title}
                subtitle={(slide as any).subtitle}
                tag={(slide as any).tag}
                tagColor={(slide as any).tagColor}
                duration={frames}
                zoomTarget={(slide as any).zoomTarget}
              />
            )}
          </Sequence>
        );
      })}
    </>
  );
};
