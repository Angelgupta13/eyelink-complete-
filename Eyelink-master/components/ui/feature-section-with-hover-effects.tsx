import { cn } from "@/lib/utils";
import {
  IconAdjustmentsBolt, // Keep if relevant, or replace
  IconHelp, // Keep if relevant, or replace
  IconRouteAltLeft, // Keep if relevant, or replace
  IconCloud, // Keep if relevant, or replace
  IconCurrencyDollar, // Keep if relevant, or replace
  IconEaseInOut, // Keep if relevant, or replace
  IconLanguage, // Example: ASL Translator icon
  IconBus, // Example: Specialized Cabs icon
  IconHeart, // Example: Care Providers icon
  IconWorldWww, // Example: Inclusivity/Connection icon
  IconAccessible, // Example: Accessibility focus icon
  IconMessageCircle, // Example: Communication icon
} from "@tabler/icons-react";

export function FeaturesSectionWithHoverEffects() {
  const features = [
    {
      title: "Real-time ASL Translation",
      description:
        "Seamlessly translate American Sign Language to text and speech, and vice versa, breaking communication barriers.",
      icon: <IconLanguage className="w-8 h-8" />,
      color: "from-blue-500/20 to-blue-500/0",
    },
    {
      title: "Accessible Cab Services",
      description:
        "Book specialized, mobility-friendly cabs for safe and convenient transportation tailored for differently-abled individuals.",
      icon: <IconBus className="w-8 h-8" />,
      color: "from-purple-500/20 to-purple-500/0",
    },
    {
      title: "Dedicated Care Provider Network",
      description:
        "Connect with vetted and compassionate care providers offering personalized support and assistance.",
      icon: <IconHeart className="w-8 h-8" />,
      color: "from-green-500/20 to-green-500/0",
    },
    {
      title: "Inclusive Communication Platform",
      description:
        "Foster genuine connections through text chat and live chat features enhanced with ASL translation.",
      icon: <IconMessageCircle className="w-8 h-8" />,
      color: "from-rose-500/20 to-rose-500/0",
    },
    {
      title: "Empowering Accessibility",
      description:
        "Eyelink is committed to making digital and physical spaces more accessible, promoting independence and inclusion.",
      icon: <IconAccessible className="w-8 h-8" />,
      color: "from-amber-500/20 to-amber-500/0",
    },
    {
      title: "Bridging Communication Gaps",
      description:
        "Our comprehensive suite of tools and services works to bridge communication gaps and build a more understanding world.",
      icon: <IconWorldWww className="w-8 h-8" />,
      color: "from-indigo-500/20 to-indigo-500/0",
    },
  ];

  return (
    <section className="py-16 sm:py-24 overflow-hidden">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
          {features.map((feature, index) => (
            <Feature key={feature.title} {...feature} index={index} />
          ))}
        </div>
      </div>
    </section>
  );
}

const Feature = ({
  title,
  description,
  icon,
  index,
  color,
}: {
  title: string;
  description: string;
  icon: React.ReactNode;
  index: number;
  color: string;
}) => {
  return (
    <div
      className={cn(
        "group/feature relative rounded-2xl p-6 transition-all duration-300",
        "hover:shadow-lg hover:-translate-y-1",
        "bg-background/50 backdrop-blur-sm",
        "border border-border/50",
        "sm:p-8"
      )}
    >
      {/* Gradient Background Effect */}
      <div
        className={cn(
          "absolute inset-0 opacity-0 group-hover/feature:opacity-100",
          "transition-opacity duration-300 rounded-2xl bg-gradient-to-b",
          color
        )}
      />

      {/* Icon Container */}
      <div className="relative mb-4 sm:mb-6">
        <div
          className={cn(
            "inline-flex items-center justify-center",
            "p-3 rounded-xl bg-background/80",
            "ring-1 ring-border/50 shadow-sm",
            "transition-transform duration-300",
            "group-hover/feature:scale-110"
          )}
        >
          {icon}
        </div>
      </div>

      {/* Content */}
      <div className="relative space-y-3">
        <h3 className="text-lg font-semibold leading-tight tracking-tight sm:text-xl">
          {title}
        </h3>
        <p className="text-sm text-muted-foreground leading-relaxed sm:text-base">
          {description}
        </p>
      </div>

      {/* Hover Indicator */}
      <div
        className={cn(
          "absolute left-0 top-8 h-12 w-1",
          "opacity-0 group-hover/feature:opacity-100",
          "transition-all duration-300",
          "bg-gradient-to-b from-rose-500 to-rose-600",
          "rounded-r-full"
        )}
      />
    </div>
  );
};
