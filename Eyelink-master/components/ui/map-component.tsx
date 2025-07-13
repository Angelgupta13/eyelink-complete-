"use client";

import React, { useEffect, useRef, useState } from "react";
import "leaflet/dist/leaflet.css";
import "leaflet-routing-machine/dist/leaflet-routing-machine.css";
import { MapPin } from "lucide-react";

interface Location {
  lat: number;
  lng: number;
  address: string;
}

// Define a new interface for the selectedLocation type
interface SelectionMode {
  type: "pickup" | "destination";
  lat: number;
  lng: number;
  address: string;
}

interface MapComponentProps {
  onLocationSelect: (location: Location) => void;
  pickup?: Location;
  destination?: Location;
  onRouteUpdate: (info: { distance: number; duration: number }) => void;
  selectedLocation?: SelectionMode | null;
}

export default function MapComponent({
  onLocationSelect,
  pickup,
  destination,
  onRouteUpdate,
  selectedLocation,
}: MapComponentProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstance = useRef<any>(null);
  const routingControlRef = useRef<any>(null);
  const [isMapReady, setIsMapReady] = useState(false);
  const [currentLocation, setCurrentLocation] = useState<{
    lat: number;
    lng: number;
  } | null>(null);
  const [userMarker, setUserMarker] = useState<any>(null);

  // Get user's current location
  useEffect(() => {
    if ("geolocation" in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setCurrentLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        (error) => {
          console.error("Error getting location:", error);
          // Default to a central India location if geolocation fails
          setCurrentLocation({ lat: 20.5937, lng: 78.9629 });
        },
        { enableHighAccuracy: true }
      );

      // Setup location watching for real-time updates
      const watchId = navigator.geolocation.watchPosition(
        (position) => {
          const newLocation = {
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          };
          setCurrentLocation(newLocation);

          // Update user marker if it exists and map is ready
          if (userMarker && mapInstance.current) {
            userMarker.setLatLng([newLocation.lat, newLocation.lng]);
          }
        },
        (error) => console.error("Error watching position:", error),
        { enableHighAccuracy: true }
      );

      // Cleanup watch on unmount
      return () => navigator.geolocation.clearWatch(watchId);
    }
  }, [userMarker]);

  useEffect(() => {
    // Load the Leaflet library
    const loadLeaflet = async () => {
      if (typeof window !== "undefined" && !window.L) {
        // Load Leaflet CSS
        const linkElem = document.createElement("link");
        linkElem.rel = "stylesheet";
        linkElem.href = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";
        document.head.appendChild(linkElem);

        // Load Leaflet JS
        const script = document.createElement("script");
        script.src = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";
        script.async = true;

        await new Promise((resolve) => {
          script.onload = resolve;
          document.head.appendChild(script);
        });

        // Load Leaflet Routing Machine
        const routingScript = document.createElement("script");
        routingScript.src =
          "https://unpkg.com/leaflet-routing-machine@3.2.12/dist/leaflet-routing-machine.js";
        routingScript.async = true;

        await new Promise((resolve) => {
          routingScript.onload = resolve;
          document.head.appendChild(routingScript);
        });
      }

      initializeMap();
    };

    loadLeaflet();

    return () => {
      if (mapInstance.current) {
        // Remove all event listeners
        mapInstance.current.off();

        // Remove routing control
        if (routingControlRef.current) {
          routingControlRef.current.remove();
        }

        // Remove map
        mapInstance.current.remove();
      }
    };
  }, [selectedLocation]);

  useEffect(() => {
    // Initialize map once current location is available
    if (currentLocation && mapInstance.current && isMapReady) {
      // Center map on current location
      mapInstance.current.setView(
        [currentLocation.lat, currentLocation.lng],
        14
      );

      // Add or update user location marker
      if (userMarker) {
        userMarker.setLatLng([currentLocation.lat, currentLocation.lng]);
      } else {
        // Create a custom user location marker
        const userIcon = window.L.divIcon({
          html: `<div class="relative">
                  <div class="w-6 h-6 bg-blue-500 rounded-full border-2 border-white"></div>
                  <div class="absolute w-12 h-12 bg-blue-500 rounded-full -top-3 -left-3 opacity-20 animate-ping"></div>
                </div>`,
          className: "user-location-marker",
          iconSize: [24, 24],
          iconAnchor: [12, 12],
        });

        const marker = window.L.marker(
          [currentLocation.lat, currentLocation.lng],
          {
            icon: userIcon,
            title: "Your Location",
            zIndexOffset: 1000,
          }
        ).addTo(mapInstance.current);

        marker.bindPopup("You are here").openPopup();
        setUserMarker(marker);
      }
    }
  }, [currentLocation, isMapReady]);

  const initializeMap = () => {
    if (!mapRef.current || !window.L) return;

    // Use current location or default to center of India
    const initialLocation = currentLocation || { lat: 20.5937, lng: 78.9629 };

    // Initialize the map
    const map = window.L.map(mapRef.current).setView(
      [initialLocation.lat, initialLocation.lng],
      currentLocation ? 14 : 5
    );

    // Add tile layer (OpenStreetMap)
    window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    }).addTo(map);

    // Add a button to center on user's location
    const locateControl = window.L.control({ position: "bottomright" });
    locateControl.onAdd = function () {
      const div = window.L.DomUtil.create("div", "leaflet-bar leaflet-control");
      div.innerHTML = `<a href="#" title="Center on your location" style="display: flex; align-items: center; justify-content: center; width: 34px; height: 34px; background: white;">
                         <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                           <circle cx="12" cy="12" r="10"></circle>
                           <circle cx="12" cy="12" r="3"></circle>
                         </svg>
                       </a>`;

      div.onclick = function (e: MouseEvent) {
        e.preventDefault();
        e.stopPropagation();
        if (currentLocation) {
          map.setView([currentLocation.lat, currentLocation.lng], 16);
        }
        return false;
      };

      return div;
    };
    locateControl.addTo(map);

    mapInstance.current = map;
    setIsMapReady(true);
  };

  // Add a separate useEffect to handle click events based on selectedLocation
  useEffect(() => {
    if (!isMapReady || !mapInstance.current) return;

    // Get map instance
    const map = mapInstance.current;

    // Remove any existing click handlers
    map.off("click");

    // Only add click handler if we're in selection mode
    if (selectedLocation) {
      map.on("click", (e: any) => {
        const { lat, lng } = e.latlng;

        // Reverse geocode to get address
        fetch(
          `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}&zoom=18&addressdetails=1`
        )
          .then((res) => res.json())
          .then((data) => {
            onLocationSelect({
              lat,
              lng,
              address: data.display_name || "Unknown location",
            });
          })
          .catch(() => {
            // Fallback if geocoding fails
            onLocationSelect({
              lat,
              lng,
              address: `${lat.toFixed(5)}, ${lng.toFixed(5)}`,
            });
          });
      });

      // Set cursor to indicate map is clickable
      if (mapRef.current) {
        mapRef.current.style.cursor = "crosshair";
      }
    } else {
      // Reset cursor when not in selection mode
      if (mapRef.current) {
        mapRef.current.style.cursor = "";
      }
    }

    // Cleanup function
    return () => {
      if (map) {
        map.off("click");
      }
    };
  }, [selectedLocation, isMapReady, onLocationSelect, currentLocation]);

  // Update markers and route when pickup/destination changes
  useEffect(() => {
    if (!isMapReady || !mapInstance.current) return;

    const map = mapInstance.current;

    // Clear existing markers and routes (except for user marker)
    map.eachLayer((layer: any) => {
      if (
        (layer instanceof window.L.Marker && layer !== userMarker) ||
        layer instanceof window.L.Polyline
      ) {
        map.removeLayer(layer);
      }
    });

    // Remove existing routing control if any
    if (routingControlRef.current) {
      routingControlRef.current.remove();
    }

    // Add the base tile layer back if it was removed
    if (!map.hasLayer(window.L.tileLayer)) {
      window.L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map);
    }

    // Add markers for pickup and destination
    const markers = [];

    // Custom icons
    const pickupIcon = window.L.divIcon({
      html: `<div class="flex items-center justify-center w-8 h-8 bg-green-500 text-white rounded-full border-2 border-white shadow-md">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <polyline points="16 12 12 8 8 12"></polyline>
                <line x1="12" y1="16" x2="12" y2="8"></line>
              </svg>
            </div>`,
      className: "pickup-marker",
      iconSize: [32, 32],
      iconAnchor: [16, 32],
    });

    const destinationIcon = window.L.divIcon({
      html: `<div class="flex items-center justify-center w-8 h-8 bg-red-500 text-white rounded-full border-2 border-white shadow-md">
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                <circle cx="12" cy="10" r="3"></circle>
              </svg>
            </div>`,
      className: "destination-marker",
      iconSize: [32, 32],
      iconAnchor: [16, 32],
    });

    if (pickup) {
      const pickupMarker = window.L.marker([pickup.lat, pickup.lng], {
        title: "Pickup",
        icon: pickupIcon,
      }).addTo(map);
      pickupMarker.bindPopup("Pickup: " + pickup.address).openPopup();
      markers.push(pickupMarker);
    }

    if (destination) {
      const destMarker = window.L.marker([destination.lat, destination.lng], {
        title: "Destination",
        icon: destinationIcon,
      }).addTo(map);
      destMarker.bindPopup("Destination: " + destination.address).openPopup();
      markers.push(destMarker);
    }

    // If both markers exist, fit the map to show both and create a route
    if (pickup && destination) {
      const group = window.L.featureGroup(markers);
      map.fitBounds(group.getBounds().pad(0.2));

      if (window.L.Routing) {
        // Create a proper route using Leaflet Routing Machine
        const routingControl = window.L.Routing.control({
          waypoints: [
            window.L.latLng(pickup.lat, pickup.lng),
            window.L.latLng(destination.lat, destination.lng),
          ],
          routeWhileDragging: false,
          showAlternatives: false,
          addWaypoints: false,
          fitSelectedRoutes: false,
          lineOptions: {
            styles: [
              { color: "black", opacity: 0.15, weight: 9 },
              { color: "#3b82f6", opacity: 0.8, weight: 6 },
            ],
          },
          createMarker: () => null, // Don't create default markers
        }).addTo(map);

        routingControlRef.current = routingControl;

        // Listen for route calculation completion
        routingControl.on("routesfound", (e: any) => {
          const routes = e.routes;
          const summary = routes[0].summary;

          // Update route info with actual distance and time
          onRouteUpdate({
            distance: summary.totalDistance / 1000, // Convert to km
            duration: Math.ceil(summary.totalTime / 60), // Convert to minutes
          });
        });
      } else {
        // Fallback to simple straight line if routing isn't available
        const latlng1 = [pickup.lat, pickup.lng];
        const latlng2 = [destination.lat, destination.lng];
        const distance = map.distance(latlng1, latlng2) / 1000; // km

        // Draw a line between the points
        const polyline = window.L.polyline([latlng1, latlng2], {
          color: "#3b82f6",
          weight: 4,
        }).addTo(map);

        // Estimate duration (average speed 30 km/h)
        const duration = Math.round((distance / 30) * 60); // minutes

        // Update route info
        onRouteUpdate({
          distance,
          duration,
        });
      }
    }
  }, [pickup, destination, isMapReady, userMarker]);

  // Use current location for pickup when the pickup button is clicked with no coordinates
  useEffect(() => {
    // Check if we have all required dependencies and we're in pickup mode with no coordinates
    if (!selectedLocation || !currentLocation || !onLocationSelect) return;

    if (
      selectedLocation.type === "pickup" &&
      selectedLocation.lat === 0 &&
      selectedLocation.lng === 0
    ) {
      // Get address for current location
      fetch(
        `https://nominatim.openstreetmap.org/reverse?format=json&lat=${currentLocation.lat}&lon=${currentLocation.lng}&zoom=18&addressdetails=1`
      )
        .then((res) => res.json())
        .then((data) => {
          onLocationSelect({
            lat: currentLocation.lat,
            lng: currentLocation.lng,
            address: data.display_name || "Your current location",
          });
        })
        .catch(() => {
          // Fallback if geocoding fails
          onLocationSelect({
            lat: currentLocation.lat,
            lng: currentLocation.lng,
            address: "Your current location",
          });
        });
    }
  }, [selectedLocation, currentLocation, onLocationSelect]);

  return (
    <div
      ref={mapRef}
      style={{ width: "100%", height: "100%", minHeight: "400px" }}
    />
  );
}

// Add this to global.d.ts or declare it here
declare global {
  interface Window {
    L: any;
  }
}
