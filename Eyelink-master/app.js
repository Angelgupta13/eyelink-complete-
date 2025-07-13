// Initialize the map
let map = L.map("map").setView([51.505, -0.09], 13); // Default view (London)

// Add OpenStreetMap tile layer
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution:
    '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

// Try to get user's location to center the map
if (navigator.geolocation) {
  navigator.geolocation.getCurrentPosition(
    function (position) {
      const lat = position.coords.latitude;
      const lon = position.coords.longitude;
      map.setView([lat, lon], 15);

      // After we get user location, query for accessibility points
      fetchAccessibilityData(lat, lon, 1); // 1km radius
    },
    function () {
      // If geolocation fails, fetch data for default location
      fetchAccessibilityData(51.505, -0.09, 2); // 2km radius
    }
  );
} else {
  // If geolocation not supported, fetch data for default location
  fetchAccessibilityData(51.505, -0.09, 2); // 2km radius
}

// Create a custom wheelchair accessible icon
function createAccessibleIcon() {
  return L.divIcon({
    className: "accessibility-icon",
    html: '<i class="fas fa-wheelchair" style="font-size: 24px; color: #1e88e5;"></i>',
    iconSize: [30, 30],
    iconAnchor: [15, 15],
    popupAnchor: [0, -15],
  });
}

// Fetch accessibility data using Overpass API
function fetchAccessibilityData(lat, lon, radius) {
  const overpassUrl = "https://overpass-api.de/api/interpreter";

  // Overpass query to find wheelchair accessible facilities
  // We're looking for nodes with wheelchair=yes or wheelchair=designated tags
  const query = `
        [out:json][timeout:25];
        (
          node["wheelchair"="yes"](around:${radius * 1000},${lat},${lon});
          node["wheelchair"="designated"](around:${
            radius * 1000
          },${lat},${lon});
          way["wheelchair"="yes"](around:${radius * 1000},${lat},${lon});
          way["wheelchair"="designated"](around:${radius * 1000},${lat},${lon});
          relation["wheelchair"="yes"](around:${radius * 1000},${lat},${lon});
          relation["wheelchair"="designated"](around:${
            radius * 1000
          },${lat},${lon});
        );
        out body;
        >;
        out skel qt;
    `;

  fetch(overpassUrl, {
    method: "POST",
    body: "data=" + encodeURIComponent(query),
  })
    .then((response) => response.json())
    .then((data) => {
      processAccessibilityData(data);
    })
    .catch((error) => {
      console.error("Error fetching accessibility data:", error);
      document.getElementById("accessibility-info").innerHTML =
        "<p>Error loading accessibility data. Please try again later.</p>";
    });
}

// Process Overpass API data and add markers to the map
function processAccessibilityData(data) {
  const accessibilityLayer = L.layerGroup().addTo(map);
  const accessibilityPoints = [];

  if (data.elements.length === 0) {
    document.getElementById("accessibility-info").innerHTML =
      "<p>No wheelchair accessibility points found in this area. You can add missing points using the button below.</p>";
    return;
  }

  // Process the data and create markers
  data.elements.forEach((element) => {
    if (element.type === "node" && element.lat && element.lon) {
      let name = element.tags.name || "Unnamed Location";
      let accessibility = "Full Access";
      if (element.tags.wheelchair === "limited") {
        accessibility = "Limited Access";
      }

      // Get additional information from tags
      const amenity = element.tags.amenity
        ? capitalizeFirstLetter(element.tags.amenity)
        : "";
      const entrance = element.tags.entrance
        ? `Entrance: ${element.tags.entrance}`
        : "";
      const description = element.tags.description || "";

      // Create a marker with custom icon
      const marker = L.marker([element.lat, element.lon], {
        icon: createAccessibleIcon(),
      }).addTo(accessibilityLayer);

      // Create popup content
      const popupContent = `
                <div class="popup-content">
                    <h3>${name}</h3>
                    <p><strong>Accessibility:</strong> ${accessibility}</p>
                    ${amenity ? `<p><strong>Type:</strong> ${amenity}</p>` : ""}
                    ${entrance ? `<p>${entrance}</p>` : ""}
                    ${description ? `<p>${description}</p>` : ""}
                    <p><strong>ID:</strong> ${element.id}</p>
                </div>
            `;

      marker.bindPopup(popupContent);

      accessibilityPoints.push({
        id: element.id,
        name: name,
        lat: element.lat,
        lon: element.lon,
        type: amenity,
        accessibility: accessibility,
      });
    }
  });

  // Update info panel
  document.getElementById("accessibility-info").innerHTML = `
        <p>Found ${accessibilityPoints.length} wheelchair accessible location(s).</p>
        <p>Click on markers to view details.</p>
    `;
}

// Helper function to capitalize first letter
function capitalizeFirstLetter(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}

// Handle adding new accessibility points
document.getElementById("add-point-btn").addEventListener("click", function () {
  enableAddPointMode();
});

function enableAddPointMode() {
  // Change the cursor to indicate add mode
  document.getElementById("map").style.cursor = "crosshair";

  // Update info panel
  document.getElementById("accessibility-info").innerHTML = `
        <p><strong>Add Mode Enabled</strong></p>
        <p>Click on the map to place a new accessibility point.</p>
        <button id="cancel-add" class="add-point-btn" style="background-color: #f44336;">Cancel</button>
    `;

  // Add cancel button functionality
  document.getElementById("cancel-add").addEventListener("click", function () {
    disableAddPointMode();
  });

  // Store the original click event
  const originalClickEvent = map.getListeners("click")[0];
  if (originalClickEvent) {
    map.off("click", originalClickEvent.fn);
  }

  // Add new click event for adding points
  map.on("click", function onMapClick(e) {
    addNewAccessibilityPoint(e.latlng);
    disableAddPointMode();

    // Restore original click event if needed
    if (originalClickEvent) {
      map.on("click", originalClickEvent.fn);
    }
  });
}

function disableAddPointMode() {
  document.getElementById("map").style.cursor = "";

  // Reset info panel
  document.getElementById("accessibility-info").innerHTML = `
        <p>Click on a marker to see accessibility details.</p>
    `;

  // Remove the click event for adding points
  map.off("click");
}

function addNewAccessibilityPoint(latlng) {
  // Create a modal form to input accessibility details
  const formHTML = `
        <div id="add-point-form" style="position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                                        z-index: 1001; background: white; padding: 20px; border-radius: 5px;
                                        box-shadow: 0 0 15px rgba(0, 0, 0, 0.3); max-width: 400px; width: 100%;">
            <h3>Add New Accessibility Point</h3>
            <form id="accessibility-form">
                <div style="margin-bottom: 15px;">
                    <label for="name" style="display: block; margin-bottom: 5px;">Name/Description:</label>
                    <input type="text" id="name" name="name" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" required>
                </div>
                <div style="margin-bottom: 15px;">
                    <label for="type" style="display: block; margin-bottom: 5px;">Type:</label>
                    <select id="type" name="type" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" required>
                        <option value="entrance">Entrance</option>
                        <option value="ramp">Ramp</option>
                        <option value="elevator">Elevator</option>
                        <option value="toilet">Accessible Toilet</option>
                        <option value="parking">Accessible Parking</option>
                        <option value="other">Other</option>
                    </select>
                </div>
                <div style="margin-bottom: 15px;">
                    <label for="access-level" style="display: block; margin-bottom: 5px;">Accessibility Level:</label>
                    <select id="access-level" name="access-level" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" required>
                        <option value="full">Full Access</option>
                        <option value="limited">Limited Access</option>
                    </select>
                </div>
                <div style="margin-bottom: 15px;">
                    <label for="notes" style="display: block; margin-bottom: 5px;">Additional Notes:</label>
                    <textarea id="notes" name="notes" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; height: 80px;"></textarea>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <button type="button" id="cancel-form" style="padding: 10px 15px; border: none; border-radius: 4px; background-color: #f44336; color: white; cursor: pointer;">Cancel</button>
                    <button type="submit" style="padding: 10px 15px; border: none; border-radius: 4px; background-color: #4CAF50; color: white; cursor: pointer;">Save Point</button>
                </div>
            </form>
        </div>
        <div id="overlay" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background-color: rgba(0, 0, 0, 0.5); z-index: 1000;"></div>
    `;

  // Add the form to the DOM
  const formContainer = document.createElement("div");
  formContainer.innerHTML = formHTML;
  document.body.appendChild(formContainer);

  // Handle cancel button
  document.getElementById("cancel-form").addEventListener("click", function () {
    document.body.removeChild(formContainer);
  });

  // Handle form submission
  document
    .getElementById("accessibility-form")
    .addEventListener("submit", function (e) {
      e.preventDefault();

      const name = document.getElementById("name").value;
      const type = document.getElementById("type").value;
      const accessLevel = document.getElementById("access-level").value;
      const notes = document.getElementById("notes").value;

      // Create a new marker
      const marker = L.marker([latlng.lat, latlng.lng], {
        icon: createAccessibleIcon(),
      }).addTo(map);

      // Create popup content
      const popupContent = `
            <div class="popup-content">
                <h3>${name}</h3>
                <p><strong>Type:</strong> ${capitalizeFirstLetter(type)}</p>
                <p><strong>Accessibility:</strong> ${
                  accessLevel === "full" ? "Full Access" : "Limited Access"
                }</p>
                ${notes ? `<p><strong>Notes:</strong> ${notes}</p>` : ""}
                <p><em>This is a user-added point (not yet in OpenStreetMap)</em></p>
            </div>
        `;

      marker.bindPopup(popupContent).openPopup();

      // In a real application, you would save this data to your database or provide
      // a mechanism to submit it to OpenStreetMap via the API
      console.log("New accessibility point:", {
        name,
        type,
        accessLevel,
        notes,
        lat: latlng.lat,
        lng: latlng.lng,
      });

      // Remove the form
      document.body.removeChild(formContainer);
    });
}
