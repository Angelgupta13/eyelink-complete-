"use client";

import { SparklesText } from "@/components/magicui/sparkles-text";
import { Card } from "@/components/ui/card";
import { Heart, Users, Globe, Shield } from "lucide-react";

const values = [
  {
    icon: Heart,
    title: "Empathy & Inclusion",
    description:
      "We believe in creating a world where everyone can communicate freely and effectively.",
  },
  {
    icon: Users,
    title: "Community First",
    description:
      "Our solutions are developed in close collaboration with the deaf and hard of hearing community.",
  },
  {
    icon: Globe,
    title: "Universal Access",
    description:
      "Making ASL translation technology accessible to everyone, everywhere.",
  },
  {
    icon: Shield,
    title: "Trust & Privacy",
    description:
      "Maintaining the highest standards of data protection and user privacy.",
  },
];

export default function AboutPage() {
  return (
    <div className="mt-24">
      <div className="relative text-center mb-16 container max-w-7xl mx-auto px-6">
        <div className="space-y-4">
          <SparklesText
            className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl"
            text="About Eyelink"
          />
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Bridging communication gaps through innovative technology
          </p>
        </div>
      </div>

      <div className="container max-w-7xl mx-auto px-6 pb-24">
        {/* Mission Statement */}
        <div className="mb-20 text-center max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold mb-6">Our Mission</h2>
          <p className="text-lg text-muted-foreground leading-relaxed">
            At Eyelink, we're dedicated to breaking down communication barriers
            between the deaf and hearing communities. Through cutting-edge AI
            technology and deep community engagement, we're making real-time ASL
            translation accessible to everyone, everywhere.
          </p>
        </div>

        {/* Values */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {values.map((value, index) => (
            <Card key={index} className="p-6 text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center mx-auto">
                <value.icon className="w-6 h-6 text-primary" />
              </div>
              <h3 className="font-semibold text-xl">{value.title}</h3>
              <p className="text-muted-foreground">{value.description}</p>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
