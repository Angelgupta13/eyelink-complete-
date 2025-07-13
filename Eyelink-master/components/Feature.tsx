import React from "react";
import { FeaturesSectionWithHoverEffects } from "@/components/ui/feature-section-with-hover-effects";

function Feature() {
  return (
    <div className="w-full relative mt-20 md:mt-32">
      <div className="w-full">
        <h1 className="text-3xl md:text-5xl font-bold text-center mb-8">
          Key Features
        </h1>
        <FeaturesSectionWithHoverEffects />
      </div>
    </div>
  );
}

export default Feature;
