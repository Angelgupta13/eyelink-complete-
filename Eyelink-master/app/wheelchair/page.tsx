"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import Head from "next/head";
import {
  fetchAccessibilityData,
  getSampleAccessibilityData,
  AccessibilityData,
  AccessibilityPoint,
  getPointTitle,
  getPointDescription,
} from "./api";
import AddAccessibilityPoint from "./components/AddAccessibilityPoint";

const WheelchairMapPage = () => {
  const mapRef = useRef<HTMLDivElement>(null);
  const [map, setMap] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [accessibilityData, setAccessibilityData] =
    useState<AccessibilityData | null>(null);
  const markersLayerRef = useRef<any>(null);
  const [addingPoint, setAddingPoint] = useState<boolean>(false);
  const [newPointPosition, setNewPointPosition] = useState<
    [number, number] | null
  >(null);
  const [userAddedPoints, setUserAddedPoints] = useState<AccessibilityPoint[]>(
    []
  );

  // Function to load and initialize the map
  const initializeMap = useCallback(async () => {
    try {
      setLoading(true);

      // Dynamically import Leaflet
      const L = await import("leaflet");
      await import("leaflet/dist/leaflet.css");

      // Return if map container is not available or map is already initialized
      if (!mapRef.current || map) return;

      // Create map instance
      const mapInstance = L.map(mapRef.current).setView([51.505, -0.09], 13);

      // Add OpenStreetMap tiles
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution:
          '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19,
      }).addTo(mapInstance);

      // Create a layer group for accessibility markers
      markersLayerRef.current = L.layerGroup().addTo(mapInstance);

      // Handle map movement/zoom to update data
      mapInstance.on("moveend", async () => {
        const bounds = mapInstance.getBounds();
        const bbox: [number, number, number, number] = [
          bounds.getSouth(),
          bounds.getWest(),
          bounds.getNorth(),
          bounds.getEast(),
        ];

        await loadAccessibilityData(bbox);
      });

      setMap(mapInstance);

      // Initial data load
      const initialBounds = mapInstance.getBounds();
      const initialBbox: [number, number, number, number] = [
        initialBounds.getSouth(),
        initialBounds.getWest(),
        initialBounds.getNorth(),
        initialBounds.getEast(),
      ];

      await loadAccessibilityData(initialBbox);
    } catch (error) {
      console.error("Error initializing map:", error);
      setError("Failed to initialize map. Please try refreshing the page.");
    } finally {
      setLoading(false);
    }
  }, [map]);

  // Function to load accessibility data
  const loadAccessibilityData = async (
    bbox: [number, number, number, number]
  ) => {
    try {
      // Use the API to fetch real data, or fall back to sample data
      let data;
      try {
        data = await fetchAccessibilityData(bbox);
        // If no points were found, use sample data
        if (data.points.length === 0) {
          data = getSampleAccessibilityData();
        }
      } catch (error) {
        console.warn(
          "Error fetching real data, using sample data instead:",
          error
        );
        data = getSampleAccessibilityData();
      }

      setAccessibilityData(data);

      // Update markers on the map if map is initialized
      if (map && markersLayerRef.current) {
        await updateMapMarkers(data);
      }
    } catch (error) {
      console.error("Error loading accessibility data:", error);
      setError("Failed to load accessibility data. Please try again later.");
    }
  };

  // Update markers on the map
  const updateMapMarkers = async (data: AccessibilityData) => {
    if (!map || !markersLayerRef.current) return;

    // Dynamically import Leaflet
    const L = await import("leaflet");

    // Clear existing markers
    markersLayerRef.current.clearLayers();

    // Create icons with different colors based on wheelchair accessibility
    const fullAccessIcon = L.icon({
      iconUrl: "/accessibility-icon.svg", // Default blue icon for full access
      iconSize: [32, 32],
      iconAnchor: [16, 32],
      popupAnchor: [0, -32],
    });

    const limitedAccessIcon = L.icon({
      iconUrl: "/accessibility-icon-limited.svg", // Yellow/orange icon for limited access
      iconSize: [32, 32],
      iconAnchor: [16, 32],
      popupAnchor: [0, -32],
    });

    const noAccessIcon = L.icon({
      iconUrl: "/accessibility-icon-no.svg", // Red icon with diagonal line for no access
      iconSize: [32, 32],
      iconAnchor: [16, 32],
      popupAnchor: [0, -32],
    });

    // Add markers for each accessibility point
    data.points.forEach((point) => {
      if (point.lat && point.lon) {
        // Choose icon based on wheelchair accessibility
        let icon = fullAccessIcon; // Default

        if (point.tags.wheelchair === "no") {
          icon = noAccessIcon;
        } else if (point.tags.wheelchair === "limited") {
          icon = limitedAccessIcon;
        }

        // Create popup content with information about the point
        const title = getPointTitle(point);
        const description = getPointDescription(point);
        const popupContent = `
          <div class="popup-content">
            <h3>${title}</h3>
            <p>${description}</p>
            ${
              point.tags.wheelchair
                ? `<div class="wheelchair-status wheelchair-${point.tags.wheelchair}">
                Wheelchair: ${point.tags.wheelchair}
              </div>`
                : ""
            }
          </div>
        `;

        // Add marker to the map
        const marker = L.marker([point.lat, point.lon], { icon })
          .bindPopup(popupContent)
          .addTo(markersLayerRef.current);
      }
    });
  };

  // Handle map click for adding new points
  const handleMapClick = useCallback(
    (e: any) => {
      if (addingPoint) {
        setNewPointPosition([e.latlng.lat, e.latlng.lng]);
      }
    },
    [addingPoint]
  );

  // Initialize map on component mount
  useEffect(() => {
    initializeMap();

    // Cleanup function
    return () => {
      if (map) {
        map.remove();
      }
    };
  }, [initializeMap]);

  // Set up map click event listener when addingPoint changes
  useEffect(() => {
    if (!map) return;

    if (addingPoint) {
      map.on("click", handleMapClick);
      // Change cursor to indicate clickable
      if (mapRef.current) {
        mapRef.current.style.cursor = "crosshair";
      }
    } else {
      map.off("click", handleMapClick);
      // Reset cursor
      if (mapRef.current) {
        mapRef.current.style.cursor = "";
      }
    }

    return () => {
      if (map) {
        map.off("click", handleMapClick);
      }
    };
  }, [map, addingPoint, handleMapClick]);

  // Update markers when accessibility data changes
  useEffect(() => {
    if (accessibilityData && map) {
      updateMapMarkers(accessibilityData);
    }
  }, [accessibilityData, map]);

  // Add a new accessibility point
  const handleAddPoint = (point: Omit<AccessibilityPoint, "id">) => {
    // Create a new point with unique ID
    const newPoint: AccessibilityPoint = {
      ...point,
      id: `user-${Date.now()}`,
    };

    // Add to user added points
    setUserAddedPoints((prev) => [...prev, newPoint]);

    // Add to the current data so it shows on the map
    setAccessibilityData((prev) => {
      if (!prev)
        return {
          points: [newPoint],
          timestamp: new Date().toISOString(),
        };

      return {
        ...prev,
        points: [...prev.points, newPoint],
      };
    });

    // Reset state
    setNewPointPosition(null);
    setAddingPoint(false);
  };

  // Cancel adding a point
  const handleCancelAddPoint = () => {
    setNewPointPosition(null);
    setAddingPoint(false);
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      <Head>
        <title>Wheelchair Accessibility Map</title>
        <meta name="description" content="Map of wheelchair ramp locations" />
      </Head>

      <header className="bg-blue-600 text-white p-4">
        <h1 className="text-2xl md:text-3xl font-bold">
          Wheelchair Accessibility Map
        </h1>
        <p className="text-sm md:text-base">
          Find wheelchair ramps and accessible entrances
        </p>
      </header>

      <main className="flex-grow p-4">
        <div className="flex justify-end mb-4">
          <button
            onClick={() => setAddingPoint(!addingPoint)}
            className={`px-4 py-2 rounded-md ${
              addingPoint
                ? "bg-red-500 text-white hover:bg-red-600"
                : "bg-blue-600 text-white hover:bg-blue-700"
            } transition-colors`}
          >
            {addingPoint ? "Cancel Adding Point" : "Add Accessibility Point"}
          </button>
        </div>

        <div className="relative bg-white dark:bg-gray-800 rounded-lg shadow-lg overflow-hidden">
          {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-100 dark:bg-gray-900 bg-opacity-80 dark:bg-opacity-80 z-10">
              <div className="flex flex-col items-center">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                <p className="mt-2 text-gray-700 dark:text-gray-300">
                  Loading map...
                </p>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-red-100 dark:bg-red-900 bg-opacity-80 dark:bg-opacity-80 z-10">
              <div className="text-center p-4">
                <p className="text-red-700 dark:text-red-300">{error}</p>
                <button
                  onClick={() => window.location.reload()}
                  className="mt-4 bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition-colors"
                >
                  Refresh
                </button>
              </div>
            </div>
          )}

          <div
            ref={mapRef}
            className="w-full h-[70vh] rounded-lg"
            aria-label="Map showing wheelchair accessibility points"
          ></div>
        </div>

        <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-2 text-gray-800 dark:text-gray-100">
              About This Map
            </h2>
            <p className="text-gray-700 dark:text-gray-400">
              This map shows wheelchair ramp locations and accessibility points.
              Click on markers to view more details about each location.
            </p>
            <p className="mt-2 text-gray-700 dark:text-gray-400">
              Data is sourced from OpenStreetMap's accessibility information.
              The map shows:
            </p>
            <ul className="list-disc ml-5 mt-2 text-gray-700 dark:text-gray-400">
              <li>Wheelchair ramps</li>
              <li>Accessible entrances</li>
              <li>Lowered kerbs</li>
              <li>Other wheelchair accessibility features</li>
            </ul>
          </div>

          <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow-md">
            <h2 className="text-lg font-semibold mb-2 text-gray-800 dark:text-gray-100">
              Legend
            </h2>
            <h3 className="text-md font-semibold mt-1 mb-2 text-gray-800 dark:text-gray-100">
              Accessibility Status:
            </h3>

            <div className="grid grid-cols-1 gap-3">
              <div className="flex items-center">
                <div
                  className="w-8 h-8 bg-contain bg-no-repeat bg-center mr-2"
                  style={{ backgroundImage: "url(/accessibility-icon.svg)" }}
                ></div>
                <span className="text-gray-700 dark:text-gray-400">
                  Full wheelchair access
                </span>
              </div>

              <div className="flex items-center">
                <div
                  className="w-8 h-8 bg-contain bg-no-repeat bg-center mr-2"
                  style={{
                    backgroundImage: "url(/accessibility-icon-limited.svg)",
                  }}
                ></div>
                <span className="text-gray-700 dark:text-gray-400">
                  Limited wheelchair access (may require assistance)
                </span>
              </div>

              <div className="flex items-center">
                <div
                  className="w-8 h-8 bg-contain bg-no-repeat bg-center mr-2"
                  style={{ backgroundImage: "url(/accessibility-icon-no.svg)" }}
                ></div>
                <span className="text-gray-700 dark:text-gray-400">
                  No wheelchair access
                </span>
              </div>
            </div>

            <h3 className="text-md font-semibold mt-4 mb-2 text-gray-800 dark:text-gray-100">
              Feature Types:
            </h3>
            <ul className="list-disc ml-5 text-gray-700 dark:text-gray-400">
              <li>Entrances with ramps</li>
              <li>Dedicated wheelchair paths</li>
              <li>Accessible facilities</li>
              <li>Lowered kerbs for street crossing</li>
            </ul>
          </div>
        </div>

        {/* Add the new component for adding points */}
        {newPointPosition && (
          <AddAccessibilityPoint
            position={newPointPosition}
            onAdd={handleAddPoint}
            onCancel={handleCancelAddPoint}
          />
        )}
      </main>

      <footer className="p-4 bg-gray-200 dark:bg-gray-700 text-center text-gray-700 dark:text-gray-300">
        <p>
          Data sourced from OpenStreetMap | &copy; {new Date().getFullYear()}
        </p>
        <p className="text-sm mt-1">
          <a
            href="https://www.openstreetmap.org/copyright"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 hover:underline dark:text-blue-400"
          >
            OpenStreetMap Contributors
          </a>
        </p>
      </footer>

      {/* Custom styles for popup content */}
      <style jsx global>{`
        .popup-content {
          padding: 5px;
          color: #333; /* Default text color for popup content */
        }
        .dark .popup-content {
          color: #eee; /* Dark mode text color for popup content */
        }

        .popup-content h3 {
          font-weight: bold;
          margin-bottom: 5px;
        }

        .wheelchair-status {
          margin-top: 5px;
          padding: 3px 6px;
          border-radius: 4px;
          display: inline-block;
          font-weight: bold;
        }

        .wheelchair-yes {
          background-color: #10b981;
          color: white;
        }

        .wheelchair-limited {
          background-color: #f59e0b;
          color: white;
        }

        .wheelchair-no {
          background-color: #ef4444;
          color: white;
        }
      `}</style>
    </div>
  );
};

export default WheelchairMapPage;
