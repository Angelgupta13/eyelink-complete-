"use client";

import React, { useEffect, useRef, useState } from "react";

interface OpenStreetMapProps {
  apiKey: string;
  onLocationSelect?: (location: {
    lat: number;
    lng: number;
    address: string;
  }) => void;
  pickup?: { lat: number; lng: number };
  destination?: { lat: number; lng: number };
  className?: string;
  onRouteUpdate?: (routeInfo: { distance: number; duration: number }) => void;
}

export function OpenStreetMap({
  apiKey,
  onLocationSelect,
  pickup,
  destination,
  className = "",
  onRouteUpdate,
}: OpenStreetMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const [map, setMap] = useState<any>(null);
  const [platform, setPlatform] = useState<any>(null);
  const [markers, setMarkers] = useState<any[]>([]);

  useEffect(() => {
    const initializeMap = () => {
      const platformObj = new (window as any).H.service.Platform({
        apikey: apiKey,
      });
      setPlatform(platformObj);

      const defaultLayers = platformObj.createDefaultLayers();
      const newMap = new (window as any).H.Map(
        mapRef.current,
        defaultLayers.vector.normal.map,
        {
          center: { lat: 20.5937, lng: 78.9629 },
          zoom: 5,
          pixelRatio: window.devicePixelRatio || 1,
        }
      );

      const events = new (window as any).H.mapevents.MapEvents(newMap);
      new (window as any).H.mapevents.Behavior(events);
      new (window as any).H.ui.UI.createDefault(newMap, defaultLayers);

      setMap(newMap);

      // Add click listener for location selection
      newMap.addEventListener("tap", (evt: any) => {
        const coord = newMap.screenToGeo(
          evt.currentPointer.viewportX,
          evt.currentPointer.viewportY
        );

        if (onLocationSelect) {
          const geocoder = platformObj.getGeocodingService();
          geocoder.reverseGeocode(
            { at: `${coord.lat},${coord.lng}` },
            (result: any) => {
              const address =
                result.items[0]?.address?.label || "Unknown address";
              onLocationSelect({
                lat: coord.lat,
                lng: coord.lng,
                address,
              });
            },
            console.error
          );
        }
      });

      return () => newMap.dispose();
    };

    if (!(window as any).H) {
      const script = document.createElement("script");
      script.src = "https://js.api.here.com/v3/3.1/mapsjs-core.js";
      script.async = true;
      script.onload = initializeMap;
      document.head.appendChild(script);

      const serviceScript = document.createElement("script");
      serviceScript.src = "https://js.api.here.com/v3/3.1/mapsjs-service.js";
      serviceScript.async = true;
      document.head.appendChild(serviceScript);
    } else {
      initializeMap();
    }
  }, [apiKey]);

  // Update markers when pickup/destination changes
  useEffect(() => {
    if (!map || !platform) return;

    // Clear existing markers
    markers.forEach((marker) => map.removeObject(marker));

    const newMarkers: any[] = [];

    if (pickup) {
      const pickupMarker = new (window as any).H.map.Marker(
        { lat: pickup.lat, lng: pickup.lng },
        { icon: new (window as any).H.map.Icon("/map-marker-red.png") }
      );
      map.addObject(pickupMarker);
      newMarkers.push(pickupMarker);
    }

    if (destination) {
      const destinationMarker = new (window as any).H.map.Marker(
        { lat: destination.lat, lng: destination.lng },
        { icon: new (window as any).H.map.Icon("/map-marker-blue.png") }
      );
      map.addObject(destinationMarker);
      newMarkers.push(destinationMarker);
    }

    setMarkers(newMarkers);

    // Calculate route if both points exist
    if (pickup && destination && onRouteUpdate) {
      const router = platform.getRoutingService();
      router.calculateRoute(
        {
          mode: "fastest;car",
          waypoint0: `geo!${pickup.lat},${pickup.lng}`,
          waypoint1: `geo!${destination.lat},${destination.lng}`,
          representation: "display",
        },
        (result: any) => {
          if (result.routes) {
            const route = result.routes[0];
            const distance = route.summary.distance / 1000; // Convert to km
            const duration = route.summary.duration / 60; // Convert to minutes
            onRouteUpdate({ distance, duration });

            // Add route to the map
            const lineString = new (window as any).H.geo.LineString();
            route.sections[0].polyline.forEach((point: string) => {
              const coords = point.split(",");
              lineString.pushLatLngAlt(
                parseFloat(coords[0]),
                parseFloat(coords[1])
              );
            });

            const routeLine = new (window as any).H.map.Polyline(lineString, {
              style: { strokeColor: "blue", lineWidth: 3 },
            });
            map.addObject(routeLine);
          }
        },
        console.error
      );
    }
  }, [pickup, destination, map, platform]);

  return <div ref={mapRef} className={`w-full h-full ${className}`} />;
}
