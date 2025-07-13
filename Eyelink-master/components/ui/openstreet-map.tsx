"use client";

import React from "react";
import dynamic from "next/dynamic";

interface Location {
  lat: number;
  lng: number;
  address: string;
}

interface SelectionMode {
  type: "pickup" | "destination";
  lat: number;
  lng: number;
  address: string;
}

interface OpenStreetMapProps {
  onLocationSelect: (location: Location) => void;
  pickup?: Location;
  destination?: Location;
  onRouteUpdate: (info: { distance: number; duration: number }) => void;
  selectedLocation?: SelectionMode | null;
}

// Create a client-side only version of the map
const MapComponent = dynamic(() => import("./map-component"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full min-h-[400px] bg-muted animate-pulse flex items-center justify-center">
      <p className="text-muted-foreground">Loading map...</p>
    </div>
  ),
});

export function OpenStreetMap(props: OpenStreetMapProps) {
  return <MapComponent {...props} />;
}
