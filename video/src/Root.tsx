import React from "react";
import { Composition } from "remotion";
import { WalkthroughComposition, TOTAL_FRAMES } from "./WalkthroughComposition";

export const Root: React.FC = () => {
  return (
    <Composition
      id="WalkthroughComposition"
      component={WalkthroughComposition}
      durationInFrames={TOTAL_FRAMES}
      fps={30}
      width={1440}
      height={900}
    />
  );
};
