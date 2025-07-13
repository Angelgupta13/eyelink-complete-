import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { MapPin } from "lucide-react";
import Link from "next/link";

export interface Provider {
  id: string;
  name: string;
  image: string;
  tagline: string;
  specializations: string[];
  location: string;
}

export function ProviderCard({ provider }: { provider: Provider }) {
  return (
    <Card className="overflow-hidden transition-all hover:shadow-lg">
      <div className="p-6 space-y-4">
        <div className="flex items-start gap-4">
          <Avatar className="h-16 w-16">
            <AvatarImage src={provider.image} alt={provider.name} />
            <AvatarFallback>{provider.name.slice(0, 2)}</AvatarFallback>
          </Avatar>
          <div className="space-y-1 flex-1">
            <h3 className="font-semibold text-lg leading-none">
              {provider.name}
            </h3>
            <div className="flex items-center text-sm text-muted-foreground">
              <MapPin className="h-4 w-4 mr-1" />
              {provider.location}
            </div>
          </div>
        </div>

        <p className="text-sm text-muted-foreground line-clamp-2">
          {provider.tagline}
        </p>

        <div className="flex flex-wrap gap-2">
          {provider.specializations.map((spec) => (
            <Badge key={spec} variant="secondary" className="font-normal">
              {spec}
            </Badge>
          ))}
        </div>

        <Link href={`/care/${provider.id}`} className="block">
          <Button className="w-full">View Profile</Button>
        </Link>
      </div>
    </Card>
  );
}

export function NoProvidersFound() {
  return (
    <Card className="p-8 text-center">
      <div className="max-w-md mx-auto space-y-4">
        <h3 className="text-lg font-semibold">No Providers Found</h3>
        <p className="text-muted-foreground">
          We couldn't find any providers matching your criteria. Try adjusting
          your filters or contact our support team for assistance.
        </p>
        <div className="flex justify-center gap-4">
          <Button variant="outline" onClick={() => window.location.reload()}>
            Reset Filters
          </Button>
          <Link href="/contact">
            <Button>Contact Support</Button>
          </Link>
        </div>
      </div>
    </Card>
  );
}
