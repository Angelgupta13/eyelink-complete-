import Feature from "@/components/Feature";

import { Hero } from "@/components/hero";
import Header from "@/components/ui/header";
import { Testimonials } from "@/components/Testinomial";

import { Pricing } from "@/components/ui/pricing";

export default function Home() {
  return (
    <>
      <Header />
      <Hero />
      <Feature />
      <Pricing
        plans={[
          {
            name: "Asl",
            price: "299",
            yearlyPrice: "199",
            period: "month",
            features: [
              "Basic ASL Translation",
              "24/7 Support",
              "Mobile Access",
            ],
            description: "Perfect for individuals starting their journey",
            buttonText: "Get Started",
            href: "/signup",
            isPopular: false,
          },
          {
            name: "Audio Navigation",
            price: "499",
            yearlyPrice: "399",
            period: "month",
            features: [
              "Real-time Audio Navigation",
              "Object & Obstacle Detection",
              "Voice Guidance",
              "Location Mapping",
            ],
            description: "Asl + Audio Navigation",
            buttonText: "Start Pro",
            href: "/signup-pro",
            isPopular: true,
          },
          {
            name: "Asl + Audio Navigation",
            price: "799",
            yearlyPrice: "699",
            period: "month",
            features: [
              "Premium ASL Translation",
              "Real-time Audio Navigation",
              "Object & Obstacle Detection",
              "Voice Guidance",
              "Location Mapping",
              "24/7 Priority Support",
              "Mobile & Desktop Access",
              "Custom ASL Signs",
              "API Integration",
            ],
            description: "For organizations and institutions",
            buttonText: "Contact Sales",
            href: "/contact",
            isPopular: false,
          },
        ]}
      />
      <Testimonials />
    </>
  );
}
