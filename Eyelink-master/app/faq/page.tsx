"use client";

import { SparklesText } from "@/components/magicui/sparkles-text";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

const faqs = [
  {
    question: "What is Eyelink ASL Translation?",
    answer:
      "Eyelink ASL Translation is an AI-powered platform that provides real-time translation of American Sign Language (ASL) to text and speech, helping bridge communication gaps between the deaf and hearing communities.",
  },
  {
    question: "How accurate is the ASL translation?",
    answer:
      "Our AI system achieves high accuracy rates and continuously improves through machine learning. However, like any translation system, accuracy may vary based on signing clarity, lighting conditions, and gesture complexity.",
  },
  {
    question: "Can I use this service on my mobile device?",
    answer:
      "Yes, Eyelink is fully responsive and works on mobile devices with cameras. You can use either the front or back camera for ASL translation.",
  },
  {
    question: "What languages are supported for translation output?",
    answer:
      "Currently, we support translation to English, Spanish, and French. We're actively working on adding more languages to our platform.",
  },
  {
    question: "Do I need special equipment for ASL translation?",
    answer:
      "No special equipment is needed. A standard webcam or mobile device camera is sufficient for using our ASL translation service.",
  },
  {
    question: "Is my data secure?",
    answer:
      "Yes, we take privacy seriously. All video streams are encrypted, processed in real-time, and not stored unless explicitly requested. We comply with GDPR and other privacy regulations.",
  },
  {
    question: "Can I save or share my translations?",
    answer:
      "Yes, you can download translations as text files or share them directly. Premium users can also access translation history and cloud storage features.",
  },
  {
    question: "What is the pricing model?",
    answer:
      "We offer various plans starting from a free tier for basic translation needs to premium plans for professional users. Check our pricing page for detailed information.",
  },
];

export default function FAQPage() {
  return (
    <div className="mt-24">
      <div className="relative text-center mb-16 container max-w-7xl mx-auto px-6">
        <div className="space-y-4">
          <SparklesText
            className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl"
            text="Frequently Asked Questions"
          />
          <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
            Find answers to common questions about Eyelink's ASL translation
            service
          </p>
        </div>
      </div>

      <div className="container max-w-4xl mx-auto px-6 pb-24">
        <Accordion type="single" collapsible className="w-full space-y-4">
          {faqs.map((faq, index) => (
            <AccordionItem key={index} value={`item-${index}`}>
              <AccordionTrigger className="text-left text-lg font-medium">
                {faq.question}
              </AccordionTrigger>
              <AccordionContent className="text-muted-foreground">
                {faq.answer}
              </AccordionContent>
            </AccordionItem>
          ))}
        </Accordion>
      </div>
    </div>
  );
}
