"use client";

import React, { useEffect, useRef, useState } from "react";
import { Loader } from "@googlemaps/js-api-loader";

interface GoogleMapProps {
  apiKey: string;
  onLocationSelect?: (location: {
    lat: number;
    lng: number;
    address: string;
  }) => void;
  pickup?: { lat: number; lng: number };
  destination?: { lat: number; lng: number };
  className?: string;
  onRouteUpdate?: (routeInfo: { distance: number; duration: number }) => void; // New prop
}

export function GoogleMap({
  apiKey,
  onLocationSelect,
  pickup,
  destination,
  className = "",
  onRouteUpdate, // Destructure the new prop
}: GoogleMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [directionsService, setDirectionsService] =
    useState<google.maps.DirectionsService | null>(null);
  const [directionsRenderer, setDirectionsRenderer] =
    useState<google.maps.DirectionsRenderer | null>(null);

  useEffect(() => {
    const loader = new Loader({
      apiKey,
      version: "weekly",
      libraries: ["places"],
    });

    loader
      .load()
      .then((google) => {
        if (mapRef.current) {
          const map = new google.maps.Map(mapRef.current, {
            center: { lat: 20.5937, lng: 78.9629 }, // Center of India
            zoom: 5,
            mapTypeControl: false,
            streetViewControl: false,
            fullscreenControl: false,
          });

          const directionsService = new google.maps.DirectionsService();
          const directionsRenderer = new google.maps.DirectionsRenderer({
            map,
            suppressMarkers: true,
          });

          setMap(map);
          setDirectionsService(directionsService);
          setDirectionsRenderer(directionsRenderer);

          // Add click event listener for location selection
          map.addListener("click", (e: google.maps.MapMouseEvent) => {
            if (e.latLng && onLocationSelect) {
              const geocoder = new google.maps.Geocoder();
              geocoder.geocode(
                { location: e.latLng },
                (
                  results: google.maps.GeocoderResult[] | null,
                  status: google.maps.GeocoderStatus
                ) => {
                  if (status === "OK" && results?.[0]) {
                    onLocationSelect({
                      lat: e.latLng?.lat() ?? 0,
                      lng: e.latLng?.lng() ?? 0,
                      address: results[0].formatted_address,
                    });
                  }
                }
              );
            }
          });
        }
      })
      .catch((error) => {
        console.error("Error loading Google Maps:", error);
      });
  }, [apiKey, onLocationSelect]); // Added onLocationSelect to dependency array

  // Update route when pickup or destination changes
  useEffect(() => {
    if (
      directionsService &&
      directionsRenderer &&
      pickup &&
      destination &&
      onRouteUpdate // Check if onRouteUpdate is provided
    ) {
      const origin = new google.maps.LatLng(pickup.lat, pickup.lng);
      const dest = new google.maps.LatLng(destination.lat, destination.lng);

      directionsService.route(
        {
          origin,
          destination: dest,
          travelMode: google.maps.TravelMode.DRIVING,
        },
        (result, status) => {
          if (status === "OK" && result) {
            directionsRenderer.setDirections(result);

            // Extract distance and duration from the route result
            const route = result.routes[0];
            if (route && route.legs && route.legs.length > 0) {
              const leg = route.legs[0];
              const distanceMeters = leg.distance?.value || 0; // Distance in meters
              const durationSeconds = leg.duration?.value || 0; // Duration in seconds

              // Convert distance to kilometers and duration to minutes
              const distanceKm = distanceMeters / 1000;
              const durationMinutes = Math.ceil(durationSeconds / 60); // Round up duration

              onRouteUpdate({
                distance: distanceKm,
                duration: durationMinutes,
              }); // Pass route info back
            } else {
              onRouteUpdate({ distance: 0, duration: 0 }); // Or handle no route found scenario
            }
          } else {
            console.error("Directions request failed:", status);
            directionsRenderer.setDirections({ routes: [] } as any); // Clear route on error
            onRouteUpdate({ distance: 0, duration: 0 }); // Inform parent component of error (optional)
          }
        }
      );
    } else {
      if (directionsRenderer) {
        directionsRenderer.setDirections({ routes: [] } as any); // Clear route if no pickup/destination
        if (onRouteUpdate) {
          onRouteUpdate({ distance: 0, duration: 0 }); // Reset fare when locations are cleared
        }
      }
    }
  }, [
    pickup,
    destination,
    directionsService,
    directionsRenderer,
    onRouteUpdate, // Added onRouteUpdate to dependency array
  ]);

  return <div ref={mapRef} className={`w-full h-full ${className}`} />;
}
