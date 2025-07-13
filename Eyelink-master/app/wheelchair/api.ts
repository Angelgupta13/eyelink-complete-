/**
 * Helper functions to fetch wheelchair accessibility data from OpenStreetMap using Overpass API
 */

// Define types for accessibility data
export interface AccessibilityPoint {
  id: string;
  type: "node" | "way" | "relation";
  lat: number;
  lon: number;
  tags: {
    name?: string;
    wheelchair?: string;
    "wheelchair:description"?: string;
    entrance?: string;
    access?: string;
    amenity?: string;
    [key: string]: string | undefined;
  };
}

export interface AccessibilityData {
  points: AccessibilityPoint[];
  timestamp: string;
  bbox?: [number, number, number, number]; // [south, west, north, east]
}

/**
 * Builds an Overpass API query to find wheelchair accessible features
 * within a bounding box
 */
const buildOverpassQuery = (bbox: [number, number, number, number]) => {
  const [south, west, north, east] = bbox;

  return `
    [out:json][timeout:25];
    (
      // Query for nodes with wheelchair tags
      node["wheelchair"](${south},${west},${north},${east});
      way["wheelchair"](${south},${west},${north},${east});
      relation["wheelchair"](${south},${west},${north},${east});
      
      // Query for entrances that might be wheelchair accessible
      node["entrance"]["wheelchair"](${south},${west},${north},${east});
      
      // Query for ramps specifically
      node["ramp"](${south},${west},${north},${east});
      way["ramp"](${south},${west},${north},${east});
      
      // Query for highways with sidewalk=both/left/right and kerb=lowered
      way["highway"]["sidewalk"]["kerb"="lowered"](${south},${west},${north},${east});
    );
    out body;
    >;
    out skel qt;
  `;
};

/**
 * Fetches wheelchair accessibility data from OpenStreetMap via Overpass API
 * for a given bounding box
 */
export const fetchAccessibilityData = async (
  bbox: [number, number, number, number]
): Promise<AccessibilityData> => {
  try {
    const query = buildOverpassQuery(bbox);
    const encodedQuery = encodeURIComponent(query);
    const response = await fetch(
      `https://overpass-api.de/api/interpreter?data=${encodedQuery}`
    );

    if (!response.ok) {
      throw new Error(`Overpass API error: ${response.status}`);
    }

    const data = await response.json();

    // Process and format the response data
    const points: AccessibilityPoint[] = data.elements
      .filter(
        (element: any) =>
          element.tags &&
          (element.tags.wheelchair ||
            element.tags.ramp ||
            (element.tags.entrance && element.tags.wheelchair) ||
            (element.tags.kerb === "lowered" && element.tags.sidewalk))
      )
      .map((element: any) => ({
        id: element.id.toString(),
        type: element.type,
        lat: element.lat || (element.center ? element.center.lat : undefined),
        lon: element.lon || (element.center ? element.center.lon : undefined),
        tags: element.tags || {},
      }))
      .filter((point: AccessibilityPoint) => point.lat && point.lon); // Filter out points without coordinates

    return {
      points,
      timestamp: data.osm3s?.timestamp_osm_base || new Date().toISOString(),
      bbox,
    };
  } catch (error) {
    console.error("Error fetching accessibility data:", error);
    // Return empty data on error
    return {
      points: [],
      timestamp: new Date().toISOString(),
      bbox,
    };
  }
};

/**
 * Returns a descriptive title for an accessibility point based on its tags
 */
export const getPointTitle = (point: AccessibilityPoint): string => {
  // Try to use the name tag first
  if (point.tags.name) {
    return point.tags.name;
  }

  // If no name, try to create a descriptive title based on available tags
  if (point.tags.amenity) {
    return (
      point.tags.amenity.charAt(0).toUpperCase() + point.tags.amenity.slice(1)
    );
  }

  if (point.tags.entrance) {
    return `${
      point.tags.entrance.charAt(0).toUpperCase() + point.tags.entrance.slice(1)
    } Entrance`;
  }

  if (point.tags.ramp) {
    return "Wheelchair Ramp";
  }

  if (point.tags.kerb === "lowered") {
    return "Lowered Kerb";
  }

  // Fallback to generic accessibility point
  return "Accessibility Point";
};

/**
 * Returns a description of the accessibility features
 */
export const getPointDescription = (point: AccessibilityPoint): string => {
  let description = "";

  // Add the wheelchair tag information
  if (point.tags.wheelchair) {
    description += `Wheelchair accessibility: ${point.tags.wheelchair}`;
  }

  // Add wheelchair description if available
  if (point.tags["wheelchair:description"]) {
    description += description ? "\n" : "";
    description += point.tags["wheelchair:description"];
  }

  // Add additional details if available
  if (point.tags.entrance) {
    description += description ? "\n" : "";
    description += `Entrance type: ${point.tags.entrance}`;
  }

  // If there's no description yet, provide a generic one
  if (!description) {
    description = "Wheelchair accessible point";
  }

  return description;
};

/**
 * Get sample accessibility data when API is not available or for testing
 */
export const getSampleAccessibilityData = (): AccessibilityData => {
  return {
    points: [
      {
        id: "1",
        type: "node",
        lat: 51.505,
        lon: -0.09,
        tags: {
          name: "Main Entrance",
          wheelchair: "yes",
          "wheelchair:description": "Wide doorway with automatic door opener",
          entrance: "main",
        },
      },
      {
        id: "2",
        type: "node",
        lat: 51.507,
        lon: -0.087,
        tags: {
          name: "Side Gate",
          wheelchair: "limited",
          "wheelchair:description": "Ramp available but steep",
          entrance: "secondary",
        },
      },
      {
        id: "3",
        type: "node",
        lat: 51.503,
        lon: -0.092,
        tags: {
          wheelchair: "yes",
          ramp: "yes",
          kerb: "lowered",
        },
      },
      {
        id: "4",
        type: "node",
        lat: 51.501,
        lon: -0.095,
        tags: {
          name: "Community Center",
          wheelchair: "yes",
          amenity: "community_centre",
          "wheelchair:description":
            "Accessible entrance with elevator to all floors",
        },
      },
    ],
    timestamp: new Date().toISOString(),
  };
};
