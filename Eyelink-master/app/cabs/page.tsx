"use client";

import React, { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Calendar } from "@/components/ui/calendar";
import { toast } from "sonner";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { format } from "date-fns";
import {
  Calendar as CalendarIcon,
  MapPin,
  Clock,
  Navigation,
} from "lucide-react";
import { SparklesText } from "@/components/magicui/sparkles-text";
import { OpenStreetMap } from "@/components/ui/openstreet-map";
import { useRouter } from "next/navigation";

// Define the SelectionMode interface
interface SelectionMode {
  type: "pickup" | "destination";
  lat: number;
  lng: number;
  address: string;
}

interface Location {
  lat: number;
  lng: number;
  address: string;
}

export default function CabBookingPage() {
  const router = useRouter();
  const [date, setDate] = useState<Date>();
  const [selectedLocation, setSelectedLocation] =
    useState<SelectionMode | null>(null);
  const [routeInfo, setRouteInfo] = useState<{
    distance: number;
    duration: number;
    fare: {
      baseFare: number;
      distanceFare: number;
      timeFare: number;
      totalFare: number;
      nightCharge?: number;
      surgeCharge?: number;
    };
  }>({
    distance: 0,
    duration: 0,
    fare: {
      baseFare: 0,
      distanceFare: 0,
      timeFare: 0,
      totalFare: 0,
    },
  });
  const [bookingData, setBookingData] = useState({
    pickup: "",
    destination: "",
    passengers: "1",
    name: "",
    phone: "",
    email: "",
    requirements: {
      wheelchair: false,
      boarding: false,
      extraSpace: false,
      serviceAnimal: false,
    },
    pickupCoords: { lat: 0, lng: 0 },
    destinationCoords: { lat: 0, lng: 0 },
  });
  const [showUseCurrentLocation, setShowUseCurrentLocation] = useState(false);
  const [hasGeolocation, setHasGeolocation] = useState(false);

  // Check if geolocation is available
  useEffect(() => {
    if ("geolocation" in navigator) {
      setHasGeolocation(true);
      // Show the current location suggestion if pickup is empty
      if (!bookingData.pickup) {
        setShowUseCurrentLocation(true);
      }
    }
  }, [bookingData.pickup]);

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setBookingData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleLocationSelect = (location: Location) => {
    if (selectedLocation) {
      const type = selectedLocation.type;
      setBookingData((prev) => ({
        ...prev,
        [type]: location.address,
        [`${type}Coords`]: { lat: location.lat, lng: location.lng },
      }));
      setSelectedLocation(null);
      setShowUseCurrentLocation(false);

      toast.success(
        `${type.charAt(0).toUpperCase() + type.slice(1)} location set!`
      );
    }
  };

  const handleRequirementChange = (requirement: string) => {
    setBookingData((prev) => ({
      ...prev,
      requirements: {
        ...prev.requirements,
        [requirement]:
          !prev.requirements[requirement as keyof typeof prev.requirements],
      },
    }));
  };

  const handleRouteUpdate = (info: { distance: number; duration: number }) => {
    const baseFare = 200;
    const perKmRate = 15;
    const perMinuteRate = 2;
    const distanceFare = Math.round(info.distance * perKmRate);
    const timeFare = Math.round(info.duration * perMinuteRate);
    const accessibilitySurcharge = Object.values(bookingData.requirements).some(
      (v) => v
    )
      ? 50
      : 0;

    setRouteInfo({
      ...info,
      fare: {
        baseFare,
        distanceFare,
        timeFare,
        totalFare: baseFare + distanceFare + timeFare + accessibilitySurcharge,
        nightCharge: undefined,
        surgeCharge: undefined,
      },
    });
  };

  const useCurrentLocation = () => {
    const selectionMode: SelectionMode = {
      type: "pickup",
      lat: 0,
      lng: 0,
      address: "",
    };
    setSelectedLocation(selectionMode);
    setShowUseCurrentLocation(false);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Validate required fields
    if (
      !bookingData.pickup ||
      !bookingData.destination ||
      !date ||
      !bookingData.name ||
      !bookingData.phone ||
      !bookingData.email
    ) {
      toast.error("Please fill in all required fields");
      return;
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(bookingData.email)) {
      toast.error("Please enter a valid email address");
      return;
    }

    // Validate phone number (basic validation for Indian numbers)
    const phoneRegex = /^[6-9]\d{9}$/;
    if (!phoneRegex.test(bookingData.phone)) {
      toast.error("Please enter a valid 10-digit phone number");
      return;
    }

    try {
      // Create booking payload
      const bookingPayload = {
        ...bookingData,
        date: date?.toISOString(),
        estimatedFare: routeInfo.fare.totalFare,
        routeInfo,
      };

      // Store booking data in localStorage for persistence
      localStorage.setItem("bookingData", JSON.stringify(bookingPayload));

      // Redirect to care page with relevant query parameters
      const queryParams = new URLSearchParams({
        pickup: bookingData.pickup,
        destination: bookingData.destination,
        date: date?.toISOString() || "",
        requirements: JSON.stringify(bookingData.requirements),
      }).toString();

      router.push(`/care?${queryParams}`);

      toast.success("Redirecting you to available care assistants...");
    } catch (error) {
      console.error("Error processing booking:", error);
      toast.error(
        "There was an error processing your request. Please try again.",
        {
          action: {
            label: "Try Again",
            onClick: () => handleSubmit(e),
          },
        }
      );
    }
  };

  // Update the map pin click handlers
  const handlePickupClick = () => {
    const selectionMode: SelectionMode = {
      type: "pickup",
      lat: 0,
      lng: 0,
      address: "",
    };
    setSelectedLocation(selectionMode);
  };

  const handleDestinationClick = () => {
    const selectionMode: SelectionMode = {
      type: "destination",
      lat: 0,
      lng: 0,
      address: "",
    };
    setSelectedLocation(selectionMode);
  };

  return (
    <div className="mt-16">
      {/* Simplified Page Header */}
      <div className="text-center mb-10 container max-w-5xl mx-auto px-6">
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl md:text-5xl">
          Book Your Assistant
        </h1>
        <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
          Safe, reliable, and accessible transportation at your service.
        </p>
      </div>

      <div className="container max-w-7xl mx-auto px-6 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Booking Form */}
          <div className="space-y-6">
            <Card className="p-6 shadow-md">
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Location Fields */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      Pickup Location
                    </label>
                    <div className="flex gap-2">
                      <Input
                        name="pickup"
                        placeholder="Enter pickup address"
                        value={bookingData.pickup}
                        onChange={handleInputChange}
                        className="h-11"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        className="flex-shrink-0"
                        onClick={handlePickupClick}
                      >
                        <MapPin className="h-5 w-5" />
                      </Button>
                    </div>
                    {showUseCurrentLocation && hasGeolocation && (
                      <Button
                        type="button"
                        variant="link"
                        className="text-primary text-sm p-0 h-auto flex gap-1 items-center"
                        onClick={useCurrentLocation}
                      >
                        <Navigation className="h-3 w-3" />
                        Use my current location
                      </Button>
                    )}
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Destination</label>
                    <div className="flex gap-2">
                      <Input
                        name="destination"
                        placeholder="Enter destination address"
                        value={bookingData.destination}
                        onChange={handleInputChange}
                        className="h-11"
                      />
                      <Button
                        type="button"
                        variant="outline"
                        className="flex-shrink-0"
                        onClick={handleDestinationClick}
                      >
                        <MapPin className="h-5 w-5" />
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Date and Time Selection */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Pickup Date</label>
                    <Popover>
                      <PopoverTrigger asChild>
                        <Button
                          variant="outline"
                          className={cn(
                            "w-full justify-start text-left font-normal h-11",
                            !date && "text-muted-foreground"
                          )}
                        >
                          <CalendarIcon className="mr-2 h-4 w-4" />
                          {date ? format(date, "PPP") : "Select date"}
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-auto p-0">
                        <Calendar
                          mode="single"
                          selected={date}
                          onSelect={setDate}
                          initialFocus
                          disabled={(date) => date < new Date()}
                        />
                      </PopoverContent>
                    </Popover>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      Number of Passengers
                    </label>
                    <Select
                      name="passengers"
                      value={bookingData.passengers}
                      onValueChange={(value) =>
                        setBookingData((prev) => ({
                          ...prev,
                          passengers: value,
                        }))
                      }
                    >
                      <SelectTrigger className="h-11">
                        <SelectValue placeholder="Select passengers" />
                      </SelectTrigger>
                      <SelectContent>
                        {[1, 2, 3, 4].map((num) => (
                          <SelectItem key={num} value={num.toString()}>
                            {num} {num === 1 ? "passenger" : "passengers"}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {/* Contact Information */}
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Full Name</label>
                    <Input
                      name="name"
                      placeholder="Enter your full name"
                      value={bookingData.name}
                      onChange={handleInputChange}
                      className="h-11"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Phone Number</label>
                    <Input
                      name="phone"
                      type="tel"
                      placeholder="Enter your phone number"
                      value={bookingData.phone}
                      onChange={handleInputChange}
                      className="h-11"
                    />
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Email Address</label>
                    <Input
                      name="email"
                      type="email"
                      placeholder="Enter your email address"
                      value={bookingData.email}
                      onChange={handleInputChange}
                      className="h-11"
                    />
                  </div>
                </div>

                {/* Accessibility Requirements */}
                <div className="space-y-4">
                  <label className="text-sm font-medium">
                    Accessibility Requirements
                  </label>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="wheelchair"
                        checked={bookingData.requirements.wheelchair}
                        onCheckedChange={() =>
                          handleRequirementChange("wheelchair")
                        }
                      />
                      <label htmlFor="wheelchair" className="text-sm">
                        Wheelchair Access
                      </label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="boarding"
                        checked={bookingData.requirements.boarding}
                        onCheckedChange={() =>
                          handleRequirementChange("boarding")
                        }
                      />
                      <label htmlFor="boarding" className="text-sm">
                        Boarding Assistance
                      </label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="extraSpace"
                        checked={bookingData.requirements.extraSpace}
                        onCheckedChange={() =>
                          handleRequirementChange("extraSpace")
                        }
                      />
                      <label htmlFor="extraSpace" className="text-sm">
                        Extra Space
                      </label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Checkbox
                        id="serviceAnimal"
                        checked={bookingData.requirements.serviceAnimal}
                        onCheckedChange={() =>
                          handleRequirementChange("serviceAnimal")
                        }
                      />
                      <label htmlFor="serviceAnimal" className="text-sm">
                        Service Animal
                      </label>
                    </div>
                  </div>
                </div>

                <Button
                  type="submit"
                  className="w-full h-11 text-base font-medium"
                  onClick={() => router.push('/care')}
                >
                  Book Assistant
                </Button>
              </form>
            </Card>
          </div>

          {/* Map Section */}
          <div className="space-y-6">
            <Card className="p-6 shadow-md">
              <div className="aspect-[4/3] bg-muted rounded-lg overflow-hidden">
                <OpenStreetMap
                  onLocationSelect={handleLocationSelect}
                  pickup={
                    bookingData.pickupCoords.lat !== 0
                      ? {
                          lat: bookingData.pickupCoords.lat,
                          lng: bookingData.pickupCoords.lng,
                          address: bookingData.pickup,
                        }
                      : undefined
                  }
                  destination={
                    bookingData.destinationCoords.lat !== 0
                      ? {
                          lat: bookingData.destinationCoords.lat,
                          lng: bookingData.destinationCoords.lng,
                          address: bookingData.destination,
                        }
                      : undefined
                  }
                  onRouteUpdate={handleRouteUpdate}
                  selectedLocation={selectedLocation}
                />
              </div>
              {selectedLocation && (
                <div className="mt-2 p-2 bg-muted rounded-md border">
                  <p className="text-sm flex items-center">
                    <MapPin className="h-4 w-4 mr-1 text-primary" />
                    {selectedLocation.lat === 0 && selectedLocation.lng === 0
                      ? `Using your current location as ${selectedLocation.type} point`
                      : `Click on the map to select ${selectedLocation.type} location`}
                  </p>
                </div>
              )}
            </Card>

            {/* Estimated Fare Card */}
            <Card className="p-6 shadow-md">
              <h3 className="font-semibold text-lg mb-4">Estimated Fare</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Base Fare</span>
                  <span className="font-medium">
                    ₹{routeInfo.fare.baseFare}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">
                    Distance Charge ({routeInfo.distance.toFixed(1)} km)
                  </span>
                  <span className="font-medium">
                    ₹{Math.round(routeInfo.fare.distanceFare)}
                  </span>
                </div>

                <div className="flex justify-between">
                  <span className="text-muted-foreground">
                    Time Charge ({routeInfo.duration} mins)
                  </span>
                  <span className="font-medium">
                    ₹{Math.round(routeInfo.fare.timeFare)}
                  </span>
                </div>

                {routeInfo.fare.nightCharge && (
                  <div className="flex justify-between text-purple-600">
                    <span>Night Charge (1.25x)</span>
                    <span className="font-medium">
                      ₹{Math.round(routeInfo.fare.nightCharge)}
                    </span>
                  </div>
                )}

                {routeInfo.fare.surgeCharge && (
                  <div className="flex justify-between text-orange-600">
                    <span>Surge Charge (1.2x)</span>
                    <span className="font-medium">
                      ₹{Math.round(routeInfo.fare.surgeCharge)}
                    </span>
                  </div>
                )}

                {Object.entries(bookingData.requirements).some(
                  ([_, value]) => value
                ) && (
                  <div className="flex justify-between text-blue-600">
                    <span>Accessibility Surcharge</span>
                    <span className="font-medium">₹50</span>
                  </div>
                )}

                <div className="pt-2 mt-2 border-t">
                  <div className="flex justify-between">
                    <span className="font-semibold">Total Estimate</span>
                    <span className="font-semibold">
                      ₹{routeInfo.fare.totalFare}
                    </span>
                  </div>
                  <div className="mt-2 space-y-1">
                    <p className="text-xs text-muted-foreground">
                      Base rate: ₹50 + ₹12/km + ₹2/min
                    </p>
                    {(new Date().getHours() >= 22 ||
                      new Date().getHours() < 5) && (
                      <p className="text-xs text-purple-600">
                        Night charges (10 PM - 5 AM): 1.25x
                      </p>
                    )}
                    {((new Date().getHours() >= 8 &&
                      new Date().getHours() <= 10) ||
                      (new Date().getHours() >= 17 &&
                        new Date().getHours() <= 20)) && (
                      <p className="text-xs text-orange-600">
                        Surge pricing (8-10 AM, 5-8 PM): 1.2x
                      </p>
                    )}
                    <p className="text-xs text-muted-foreground">
                      Minimum fare: ₹100
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Final fare may vary based on actual route and waiting time
                    </p>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
