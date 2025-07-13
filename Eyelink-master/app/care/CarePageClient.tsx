"use client";

import React from "react";
import { SparklesText } from "@/components/magicui/sparkles-text";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import Link from "next/link";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Search, MapPin, Clock, Languages, Star, Award } from "lucide-react";
import takerData from "@/data/taker.json";

interface Review {
  author: string;
  rating: number;
  text: string;
}

interface Provider {
  id: string;
  name: string;
  profilePicture: string;
  bio: string;
  location: string;
  specializations: string[];
  languages: string[];
  availability: string;
  hourlyRate: number;
  reviews: Review[];
}

const specializations = [
  { id: "asl", label: "ASL Communication" },
  { id: "mobility", label: "Mobility Assistance" },
  { id: "personal", label: "Personal Care" },
  { id: "medical", label: "Medical Assistance" },
  { id: "companionship", label: "Companionship" },
  { id: "respite", label: "Respite Care" },
];

const languages = [
  { value: "en", label: "English" },
  { value: "asl", label: "ASL" },
  { value: "es", label: "Spanish" },
  { value: "fr", label: "French" },
];

const availability = [
  { value: "weekdays", label: "Weekdays" },
  { value: "weekends", label: "Weekends" },
  { value: "evenings", label: "Evenings" },
  { value: "flexible", label: "Flexible" },
];

interface CarePageClientProps {
  initialSearchParams: { [key: string]: string | string[] | undefined };
}

export default function CarePageClient({
  initialSearchParams,
}: CarePageClientProps) {
  const [providers, setProviders] = React.useState<Provider[]>(
    takerData as Provider[]
  );
  const [bookingDetails, setBookingDetails] = React.useState<any>(null);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedLocation, setSelectedLocation] = React.useState("");
  const [selectedAvailability, setSelectedAvailability] = React.useState("");
  const [selectedLanguage, setSelectedLanguage] = React.useState("");
  const [selectedPriceRange, setSelectedPriceRange] = React.useState("");
  const [selectedSpecializations, setSelectedSpecializations] = React.useState<
    string[]
  >([]);

  React.useEffect(() => {
    // Get booking details from URL params and localStorage
    const storedBooking = localStorage.getItem("bookingData");
    const bookingData = storedBooking ? JSON.parse(storedBooking) : null;

    if (bookingData || Object.keys(initialSearchParams).length > 0) {
      setBookingDetails({
        ...bookingData,
        pickup: initialSearchParams.pickup || bookingData?.pickup,
        destination:
          initialSearchParams.destination || bookingData?.destination,
        date: initialSearchParams.date || bookingData?.date,
        requirements: initialSearchParams.requirements
          ? JSON.parse(initialSearchParams.requirements as string)
          : bookingData?.requirements,
      });

      // Filter providers based on requirements
      const requirements = initialSearchParams.requirements
        ? JSON.parse(initialSearchParams.requirements as string)
        : bookingData?.requirements;

      if (requirements) {
        const filteredProviders = takerData.filter((provider) => {
          return provider.specializations.some((spec) =>
            requirements.boarding
              ? spec.includes("Mobility")
              : true && requirements.extraSpace
              ? spec.includes("Special")
              : true && requirements.serviceAnimal
              ? spec.includes("Animal")
              : true
          );
        });
        setProviders(filteredProviders);
      }
    }
  }, [initialSearchParams]);

  const handleSearch = () => {
    // Filter providers based on search criteria
    const filteredProviders = (takerData as Provider[]).filter((provider) => {
      const matchesSearch =
        searchQuery === "" ||
        provider.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        provider.specializations.some((s) =>
          s.toLowerCase().includes(searchQuery.toLowerCase())
        );

      const matchesLocation =
        selectedLocation === "" ||
        provider.location.toLowerCase() === selectedLocation.toLowerCase();

      const matchesAvailability =
        selectedAvailability === "" ||
        provider.availability.toLowerCase() ===
          selectedAvailability.toLowerCase();

      const matchesLanguage =
        selectedLanguage === "" ||
        provider.languages.some(
          (l) => l.toLowerCase() === selectedLanguage.toLowerCase()
        );

      const matchesPriceRange =
        selectedPriceRange === "" ||
        matchesPriceRangeFilter(provider.hourlyRate, selectedPriceRange);

      const matchesSpecializations =
        selectedSpecializations.length === 0 ||
        selectedSpecializations.every((s) =>
          provider.specializations.includes(s)
        );

      return (
        matchesSearch &&
        matchesLocation &&
        matchesAvailability &&
        matchesLanguage &&
        matchesPriceRange &&
        matchesSpecializations
      );
    });

    setProviders(filteredProviders);
  };

  const clearFilters = () => {
    setSearchQuery("");
    setSelectedLocation("");
    setSelectedAvailability("");
    setSelectedLanguage("");
    setSelectedPriceRange("");
    setSelectedSpecializations([]);
    setProviders(takerData as Provider[]);
  };

  // Helper function to match price range
  const matchesPriceRangeFilter = (rate: number, range: string) => {
    const [min, max] = range.split("-").map(Number);
    if (range === "75+") return rate >= 75;
    return rate >= min && rate <= max;
  };

  React.useEffect(() => {
    handleSearch();
  }, [searchQuery, selectedLocation, selectedSpecializations]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20 pt-24">
      {/* Page Header */}
      <div className="relative text-center mb-16 container max-w-7xl mx-auto px-6">
        <div className="space-y-12">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            {bookingDetails ? "Recommended Assistants" : "Our Care Assistants"}
          </h1>
          
          {bookingDetails && (
            <div className="flex flex-wrap gap-4 justify-center items-center">
              <Badge variant="secondary" className="px-4 py-2 text-base">
                <MapPin className="w-4 h-4 mr-2" />
                {bookingDetails.pickup} → {bookingDetails.destination}
              </Badge>
              <Badge variant="secondary" className="px-4 py-2 text-base">
                <Clock className="w-4 h-4 mr-2" />
                {new Date(bookingDetails.date).toLocaleDateString()}
              </Badge>
            </div>
          )}
          <p className="text-lg sm:text-xl text-muted-foreground max-w-3xl mx-auto">
            {bookingDetails
              ? "Based on your requirements, we've found these qualified assistants for your journey."
              : "Our experienced and verified care assistants are here to help you with your needs."}
          </p>
        </div>
      </div>

      {/* Providers Grid */}
      <div className="container max-w-7xl mx-auto px-6 pb-20">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {providers.map((provider) => {
            const averageRating =
              provider.reviews.reduce((acc, review) => acc + review.rating, 0) /
              provider.reviews.length;

            return (
              <Link
                key={provider.id}
                href={`/care/${provider.id}`}
                className="transition-transform hover:scale-[1.02]"
              >
                <Card className="p-6 h-full flex flex-col">
                  <div className="flex items-start gap-4 mb-4">
                    <Avatar className="h-16 w-16">
                      <AvatarImage
                        src={provider.profilePicture}
                        alt={provider.name}
                      />
                      <AvatarFallback>
                        {provider.name.slice(0, 2)}
                      </AvatarFallback>
                    </Avatar>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold">{provider.name}</h3>
                      <div className="flex items-center gap-2 text-muted-foreground text-sm">
                        <MapPin className="h-4 w-4" />
                        <span>{provider.location}</span>
                      </div>
                      <div className="flex items-center gap-1 mt-1">
                        <Star className="h-4 w-4 fill-primary text-primary" />
                        <span className="font-medium">
                          {averageRating.toFixed(1)}
                        </span>
                        <span className="text-muted-foreground">
                          ({provider.reviews.length})
                        </span>
                      </div>
                    </div>
                  </div>

                  <p className="text-muted-foreground mb-4 line-clamp-2">
                    {provider.bio}
                  </p>

                  <div className="mt-auto">
                    <div className="flex flex-wrap gap-2 mb-4">
                      {provider.specializations.slice(0, 3).map((spec) => (
                        <Badge
                          key={spec}
                          variant="secondary"
                          className="px-2 py-1 text-xs"
                        >
                          {spec}
                        </Badge>
                      ))}
                      {provider.specializations.length > 3 && (
                        <Badge
                          variant="secondary"
                          className="px-2 py-1 text-xs"
                        >
                          +{provider.specializations.length - 3} more
                        </Badge>
                      )}
                    </div>

                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="px-3 py-1">
                        <Clock className="w-4 h-4 mr-1" />
                        {provider.availability}
                      </Badge>
                      <Badge
                        variant="outline"
                        className="px-3 py-1 bg-primary/5"
                      >
                        <Award className="w-4 h-4 mr-1" />${provider.hourlyRate}
                        /hr
                      </Badge>
                    </div>
                  </div>
                </Card>
              </Link>
            );
          })}
        </div>
      </div>
    </div>
  );
}
