"use client";

import React from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  Star,
  MapPin,
  Clock,
  Languages,
  Mail,
  Calendar,
  Shield,
  Heart,
  Award,
} from "lucide-react";
import takerData from "@/data/taker.json";
import { notFound } from "next/navigation";

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

type PageProps = {
  params: Promise<{ provider: string }>;
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
};

export default function ProviderPage({ params, searchParams }: PageProps) {
  const [provider, setProvider] = React.useState<Provider | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    async function loadProvider() {
      try {
        const [{ provider: providerId }, queryParams] = await Promise.all([
          params,
          searchParams,
        ]);
        const providerData = takerData.find((p) => p.id === providerId);

        if (!providerData) {
          setError("Provider not found");
          return;
        }

        setProvider(providerData);
      } catch (err) {
        setError("Failed to load provider data");
      } finally {
        setLoading(false);
      }
    }

    loadProvider();
  }, [params, searchParams]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-pulse text-lg text-muted-foreground">
          Loading provider details...
        </div>
      </div>
    );
  }

  if (error || !provider) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Card className="p-6">
          <h2 className="text-xl font-semibold text-red-500 mb-2">Error</h2>
          <p className="text-muted-foreground">
            {error || "Provider not found"}
          </p>
          <Button className="mt-4" onClick={() => window.history.back()}>
            Go Back
          </Button>
        </Card>
      </div>
    );
  }

  const averageRating =
    provider.reviews.reduce((acc, review) => acc + review.rating, 0) /
    provider.reviews.length;

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20 pt-24">
      {/* Hero Section */}
      <div className="relative bg-primary/5 py-12 overflow-hidden mb-8">
        <div className="absolute inset-0 bg-grid-white/10" />
        <div className="container mx-auto px-4">
          <div className="relative bg-background/60 backdrop-blur-lg rounded-2xl p-8 shadow-lg">
            <div className="flex flex-col md:flex-row gap-8 items-start">
              {/* Profile Picture Section */}
              <div className="flex-shrink-0">
                <div className="relative">
                  <Avatar className="h-40 w-40 ring-4 ring-primary/10">
                    <AvatarImage
                      src={provider.profilePicture}
                      alt={provider.name}
                      className="object-cover"
                    />
                    <AvatarFallback className="text-2xl">
                      {provider.name.slice(0, 2)}
                    </AvatarFallback>
                  </Avatar>
                  <Badge className="absolute -bottom-2 right-0 px-4 py-1">
                    <Shield className="w-4 h-4 mr-1" /> Verified
                  </Badge>
                </div>
              </div>

              {/* Provider Info */}
              <div className="flex-grow space-y-6">
                <div className="flex flex-col md:flex-row justify-between gap-4">
                  <div className="space-y-2">
                    <h1 className="text-4xl font-bold tracking-tight">
                      {provider.name}
                    </h1>
                    <div className="flex items-center gap-4 text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <MapPin className="h-4 w-4" />
                        <span>{provider.location}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Star className="h-4 w-4 fill-primary text-primary" />
                        <span>
                          {averageRating.toFixed(1)} ({provider.reviews.length}{" "}
                          reviews)
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <Button size="lg" className="shadow-lg">
                      <Mail className="mr-2 h-5 w-5" />
                      Contact Now
                    </Button>
                    <Button size="lg" variant="outline">
                      <Calendar className="mr-2 h-5 w-5" />
                      Schedule Call
                    </Button>
                  </div>
                </div>

                <div className="prose prose-sm max-w-none">
                  <p className="text-lg text-muted-foreground leading-relaxed">
                    {provider.bio}
                  </p>
                </div>

                <div className="flex flex-wrap gap-6">
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="px-3 py-1">
                      <Clock className="w-4 h-4 mr-1" />
                      {provider.availability}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="px-3 py-1">
                      <Languages className="w-4 h-4 mr-1" />
                      {provider.languages.join(", ")}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="px-3 py-1 bg-primary/5">
                      <Award className="w-4 h-4 mr-1" />${provider.hourlyRate}
                      /hr
                    </Badge>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Content Section */}
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Left Column */}
          <div className="md:col-span-2 space-y-8">
            {/* Specializations */}
            <Card className="p-6">
              <h2 className="text-2xl font-semibold mb-4">
                Expertise & Specializations
              </h2>
              <div className="flex flex-wrap gap-2">
                {provider.specializations.map((spec) => (
                  <Badge
                    key={spec}
                    variant="secondary"
                    className="px-3 py-1.5 text-sm"
                  >
                    {spec}
                  </Badge>
                ))}
              </div>
            </Card>

            {/* Reviews */}
            <Card className="p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-semibold">Client Reviews</h2>
                <Badge variant="secondary" className="px-3 py-1">
                  {provider.reviews.length} Reviews
                </Badge>
              </div>

              <div className="space-y-6">
                {provider.reviews.length > 0 ? (
                  provider.reviews.map((review, index) => (
                    <div
                      key={index}
                      className="bg-muted/50 rounded-lg p-4 transition-colors hover:bg-muted/70"
                    >
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex items-center gap-2">
                          <Avatar className="h-8 w-8">
                            <AvatarFallback>
                              {review.author.slice(0, 2)}
                            </AvatarFallback>
                          </Avatar>
                          <span className="font-medium">{review.author}</span>
                        </div>
                        <div className="flex items-center">
                          {Array.from({ length: review.rating }).map((_, i) => (
                            <Star
                              key={i}
                              className="h-4 w-4 fill-primary text-primary"
                            />
                          ))}
                        </div>
                      </div>
                      <p className="text-muted-foreground">{review.text}</p>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Heart className="h-12 w-12 mx-auto mb-3 opacity-20" />
                    <p>No reviews yet. Be the first to review!</p>
                  </div>
                )}
              </div>
            </Card>
          </div>

          {/* Right Column - Booking Card */}
          <div className="md:col-span-1">
            <Card className="p-6 sticky top-6">
              <h3 className="text-xl font-semibold mb-4">Book a Session</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-muted-foreground">Hourly Rate</span>
                  <span className="font-semibold">
                    ${provider.hourlyRate}/hr
                  </span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-muted-foreground">Availability</span>
                  <span className="font-medium">{provider.availability}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b">
                  <span className="text-muted-foreground">Response Time</span>
                  <span className="font-medium">Within 24 hours</span>
                </div>
                <Button className="w-full" size="lg">
                  Book Now
                </Button>
                <p className="text-xs text-muted-foreground text-center mt-2">
                  Free cancellation up to 24 hours before the session
                </p>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
