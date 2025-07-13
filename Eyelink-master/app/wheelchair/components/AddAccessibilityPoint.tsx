"use client";

import { useState } from "react";
import { AccessibilityPoint } from "../api";

interface AddAccessibilityPointProps {
  onAdd: (point: Omit<AccessibilityPoint, "id">) => void;
  position: [number, number] | null;
  onCancel: () => void;
}

const AddAccessibilityPoint: React.FC<AddAccessibilityPointProps> = ({
  onAdd,
  position,
  onCancel,
}) => {
  const [name, setName] = useState("");
  const [accessibilityStatus, setAccessibilityStatus] = useState<
    "yes" | "limited" | "no"
  >("yes");
  const [description, setDescription] = useState("");
  const [entranceType, setEntranceType] = useState<string>("");
  const [hasRamp, setHasRamp] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!position) {
    return null;
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    setIsSubmitting(true);

    // Create a new accessibility point
    const newPoint: Omit<AccessibilityPoint, "id"> = {
      type: "node",
      lat: position[0],
      lon: position[1],
      tags: {
        name: name || undefined,
        wheelchair: accessibilityStatus,
        "wheelchair:description": description || undefined,
        entrance: entranceType || undefined,
        ramp: hasRamp ? "yes" : undefined,
      },
    };

    // Pass the new point to the parent component
    onAdd(newPoint);

    // Reset form
    setName("");
    setAccessibilityStatus("yes");
    setDescription("");
    setEntranceType("");
    setHasRamp(false);
    setIsSubmitting(false);
  };

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-20">
      <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full max-h-[90vh] overflow-y-auto">
        <h2 className="text-xl font-bold mb-4">Add Accessibility Point</h2>
        <p className="text-gray-600 mb-4">
          At position: {position[0].toFixed(6)}, {position[1].toFixed(6)}
        </p>

        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label
              htmlFor="name"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Location Name
            </label>
            <input
              type="text"
              id="name"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Main Entrance, South Gate"
            />
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Wheelchair Accessibility
            </label>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  name="accessibility"
                  checked={accessibilityStatus === "yes"}
                  onChange={() => setAccessibilityStatus("yes")}
                  className="mr-2"
                />
                <span>Full Access</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="accessibility"
                  checked={accessibilityStatus === "limited"}
                  onChange={() => setAccessibilityStatus("limited")}
                  className="mr-2"
                />
                <span>Limited</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="accessibility"
                  checked={accessibilityStatus === "no"}
                  onChange={() => setAccessibilityStatus("no")}
                  className="mr-2"
                />
                <span>No Access</span>
              </label>
            </div>
          </div>

          <div className="mb-4">
            <label
              htmlFor="description"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Description
            </label>
            <textarea
              id="description"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
              placeholder="Describe the accessibility features or limitations"
            />
          </div>

          <div className="mb-4">
            <label
              htmlFor="entranceType"
              className="block text-sm font-medium text-gray-700 mb-1"
            >
              Entrance Type
            </label>
            <select
              id="entranceType"
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              value={entranceType}
              onChange={(e) => setEntranceType(e.target.value)}
            >
              <option value="">Not an entrance</option>
              <option value="main">Main Entrance</option>
              <option value="secondary">Secondary Entrance</option>
              <option value="emergency">Emergency Exit</option>
              <option value="service">Service Entrance</option>
            </select>
          </div>

          <div className="mb-4">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={hasRamp}
                onChange={(e) => setHasRamp(e.target.checked)}
                className="mr-2"
              />
              <span className="text-sm font-medium text-gray-700">
                Has Wheelchair Ramp
              </span>
            </label>
          </div>

          <div className="flex justify-end space-x-3">
            <button
              type="button"
              className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-gray-500"
              onClick={onCancel}
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isSubmitting}
            >
              {isSubmitting ? "Adding..." : "Add Point"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AddAccessibilityPoint;
