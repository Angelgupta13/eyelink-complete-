"use client";

import { Pricing } from "@/components/ui/pricing";

export default function PricingPage() {
  return (
    <Pricing
      title="Simple, Transparent Pricing"
      description="Choose the plan that works best for you\nAll plans include access to our ASL translation platform and dedicated support."
      plans={[
        {
          name: "Asl",
          price: "299",
          yearlyPrice: "199",
          period: "month",
          features: ["Basic ASL Translation", "24/7 Support", "Mobile Access"],
          description:
            "Perfect for individuals starting their journey with ASL",
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
          description: "Ideal for active learners and professionals",
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
          description:
            "For organizations and institutions requiring full features",
          buttonText: "Contact Sales",
          href: "/contact",
          isPopular: false,
        },
      ]}
    />
  );
}
