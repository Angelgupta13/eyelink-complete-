import { TestimonialsSection } from "@/components/ui/testinomail-marique";

const testimonials = [
  {
    author: {
      name: "Aisha Patel",
      handle: "@aishasign", // Example handle, can be removed if not needed
      avatar:
        "https://images.unsplash.com/photo-1595152778340-res/unsplash-photo-1595152778340-res?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8NXx8d29tYW58ZW58MHx8MHx8&auto=format&fit=crop&w=150&h=150&q=60", // Placeholder image - suggest replacing with more diverse and relevant images
    },
    text: "As someone who is deaf, Eyelink's ASL Translator has been a revelation.  Suddenly, conversations that used to be a struggle are now smooth and natural. It's truly bridging communication gaps and making me feel more connected.",
    // href: "https://twitter.com/aishasign", // Removed href as these are likely fictional
  },
  {
    author: {
      name: "Carlos Rodriguez",
      handle: "@carlosmobility", // Example handle, can be removed if not needed
      avatar:
        "https://images.unsplash.com/photo-1580489944761-15a19d674b80?w=150&h=150&fit=crop&crop=face", // Placeholder image - suggest replacing with more diverse and relevant images
    },
    text: "Getting around used to be a major challenge. Eyelink's specialized cab service is a game-changer.  Booking is easy, the cabs are comfortable and accessible, and it's given me back so much independence.  Thank you, Eyelink!",
    // href: "https://twitter.com/carlosmobility", // Removed href as these are likely fictional
  },
  {
    author: {
      name: "Mei Ling",
      handle: "@meicaregiver", // Example handle, can be removed if not needed
      avatar:
        "https://images.unsplash.com/photo-1581093456484-a4d44388a759?w=150&h=150&fit=crop&crop=face", // Placeholder image - suggest replacing with more diverse and relevant images
    },
    text: "Finding reliable and understanding care for my mother has been difficult. Eyelink's care provider platform connected me with Sarah, who is not only skilled but also genuinely caring and patient.  It's brought peace of mind to our family.",
  },
  {
    author: {
      name: "David Chen",
      handle: "@davidtech",
      avatar:
        "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=150&h=150&fit=crop&crop=face",
    },
    text: "The real-time ASL translation technology has transformed how I teach my online classes. My deaf students can now participate fully in discussions, making education truly inclusive.",
  },
  {
    author: {
      name: "Sarah Thompson",
      handle: "@sarahhealth",
      avatar:
        "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e?w=150&h=150&fit=crop&crop=face",
    },
    text: "As a healthcare provider, Eyelink's platform has helped me better serve my deaf patients. The seamless communication makes medical consultations more accurate and comfortable for everyone.",
  },
  {
    author: {
      name: "Marcus Johnson",
      handle: "@marcusj",
      avatar:
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face",
    },
    text: "The emergency response feature is incredible. When I needed urgent medical attention, Eyelink's quick connection to ASL-fluent emergency services made a stressful situation much more manageable.",
  },
];

export function Testimonials() {
  return (
    <TestimonialsSection
      title="Trusted by our community of users and care providers"
      description="Join thousands who are breaking down communication barriers and accessing better care with Eyelink"
      testimonials={testimonials}
    />
  );
}
