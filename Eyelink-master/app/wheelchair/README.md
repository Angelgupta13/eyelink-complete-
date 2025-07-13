# Wheelchair Accessibility Map

A web application that displays wheelchair ramp locations and accessibility features using OpenStreetMap and Leaflet.js.

## Features

- Interactive map powered by OpenStreetMap and Leaflet.js
- Displays wheelchair accessibility points including:
  - Entrances with ramps
  - Accessible buildings and facilities
  - Lowered kerbs for street crossing
  - Other accessibility features
- Color-coded accessibility icons indicating:
  - Full wheelchair access (blue)
  - Limited wheelchair access (yellow/orange)
  - No wheelchair access (red)
- Detailed pop-up information for each accessibility point
- Search capability by location
- Ability to add missing accessibility points to the map
- Responsive design for desktop and mobile devices

## Data Sources

The application retrieves wheelchair accessibility data from OpenStreetMap using the Overpass API. It queries for:

- Nodes, ways, and relations with wheelchair tags
- Entrances with wheelchair accessibility information
- Dedicated ramps
- Highways with lowered kerbs and sidewalks

## Technical Implementation

This application is built with:

- Next.js 13+ (App Router)
- React 18+
- TypeScript
- Leaflet.js for mapping functionality
- Tailwind CSS for styling
- Overpass API for OpenStreetMap data

## Components

- `app/wheelchair/page.tsx`: Main page component that renders the map
- `app/wheelchair/api.ts`: Helper functions to fetch and process accessibility data
- `app/wheelchair/components/AddAccessibilityPoint.tsx`: Component for adding new accessibility points

## Icons

The application uses custom SVG icons to represent different accessibility states:

- `public/accessibility-icon.svg`: Full wheelchair access (blue)
- `public/accessibility-icon-limited.svg`: Limited wheelchair access (yellow/orange)
- `public/accessibility-icon-no.svg`: No wheelchair access (red)

## How to Use

1. **Viewing Accessibility Points**
   - Browse the map to see accessibility points marked with wheelchair icons
   - Click on any marker to view detailed information
   - Different colored icons indicate different levels of accessibility

2. **Adding Missing Accessibility Points**
   - Click the "Add Accessibility Point" button
   - Click on the map where you want to add a point
   - Fill in the form with accessibility details
   - Submit to add the point to the map

3. **Navigating the Map**
   - Zoom in/out using the +/- controls or mouse wheel
   - Pan by clicking and dragging the map
   - The map automatically loads new accessibility data as you navigate

## Future Enhancements

- User authentication for contributing data
- Ability to edit and update existing accessibility points
- Integration with OSM API to contribute data back to OpenStreetMap
- Routing functionality for accessible paths
- Filter options to show specific types of accessibility features
- Image upload for accessibility points
- Offline support for map data

## Accessibility

This application is designed with accessibility in mind:

- Semantic HTML structure
- Keyboard navigation support
- Screen reader friendly elements
- ARIA attributes where appropriate
- Color contrast compliant design

## License

This project uses data from OpenStreetMap, which is available under the Open Database License (ODbL).
