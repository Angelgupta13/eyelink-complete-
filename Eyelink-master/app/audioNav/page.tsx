"use client";

import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  ArrowRight,
  Camera,
  Cpu,
  CloudUpload,
  Headphones,
  MapPin,
  Settings,
  AlertTriangle,
  Shield,
  Zap,
  Building,
  Bus,
} from "lucide-react";
import Image from "next/image";
import { cn } from "@/lib/utils";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";

const steps = [
  {
    id: 1,
    title: "Smart Camera Sensing",
    description:
      "Ultra-compact cameras embedded in AudioNav earbuds capture a continuous visual feed of your surroundings in real-time.",
    icon: <Camera className="h-10 w-10 text-primary" />,
    color: "bg-blue-100",
    details:
      "Our miniaturized camera technology is practically invisible to others, built directly into our specially designed earbuds. The cameras utilize wide-angle lenses to maximize visual coverage while maintaining high resolution even in low-light conditions.",
  },
  {
    id: 2,
    title: "On-Device Processing",
    description:
      "Initial object detection happens instantly on the device itself, identifying immediate obstacles without delay.",
    icon: <Cpu className="h-10 w-10 text-primary" />,
    color: "bg-green-100",
    details:
      "AudioNav uses a lightweight but powerful neural network for immediate object detection right on the device. This ensures critical safety features like obstacle detection work even without internet connectivity and with minimal latency.",
  },
  {
    id: 3,
    title: "Cloud Intelligence",
    description:
      "Encrypted image data is securely transmitted to our cloud servers for advanced environmental analysis.",
    icon: <CloudUpload className="h-10 w-10 text-primary" />,
    color: "bg-purple-100",
    details:
      "For more complex scene understanding, encrypted visual data is sent to our secure cloud infrastructure. Here, our advanced AI models can identify specific landmarks, read text, recognize faces of saved contacts, and understand spatial contexts that require more computing power.",
  },
  {
    id: 4,
    title: "Contextual Audio Guidance",
    description:
      "Natural, conversational audio cues are delivered through your earbuds, describing your surroundings and providing navigation.",
    icon: <Headphones className="h-10 w-10 text-primary" />,
    color: "bg-orange-100",
    details:
      "Rather than robotic instructions, AudioNav delivers human-like guidance that prioritizes what's most important in your environment. The system uses 3D audio to provide directional cues, making it intuitive to understand where objects are located relative to your position.",
  },
  {
    id: 5,
    title: "Spatial Mapping",
    description:
      "The system builds a personalized map of frequented locations, improving navigation accuracy over time.",
    icon: <MapPin className="h-10 w-10 text-primary" />,
    color: "bg-red-100",
    details:
      "As you use AudioNav, it learns your common paths and destinations. This allows the system to provide increasingly accurate guidance and remember important landmarks in your regular environments, creating a personalized spatial map that enhances navigation reliability.",
  },
];

const features = [
  {
    title: "Privacy-First Design",
    description:
      "All visual data is processed with privacy as the priority. Images are never stored longer than needed for processing, and biometric encryption keeps your data secure.",
    icon: <Shield className="h-6 w-6 text-green-500" />,
  },
  {
    title: "Ultra-Low Latency",
    description:
      "Critical navigation cues are delivered in under 100ms, providing real-time guidance without noticeable delay.",
    icon: <Zap className="h-6 w-6 text-yellow-500" />,
  },
  {
    title: "Customizable Experience",
    description:
      "Adjust verbosity, detail level, and voice characteristics to create your perfect navigation assistant.",
    icon: <Settings className="h-6 w-6 text-blue-500" />,
  },
  {
    title: "Offline Capabilities",
    description:
      "Core navigation features function even without internet connection, ensuring reliability anywhere.",
    icon: <AlertTriangle className="h-6 w-6 text-orange-500" />,
  },
];

export default function AudioNavigationPage() {
  const [activeStep, setActiveStep] = useState(1);
  const [isVisible, setIsVisible] = useState<Record<string, boolean>>({});
  const { theme } = useTheme();

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setIsVisible((prev) => ({
              ...prev,
              [entry.target.id]: true,
            }));
          }
        });
      },
      { threshold: 0.3 }
    );

    document.querySelectorAll(".observe-me").forEach((el) => {
      observer.observe(el);
    });

    return () => {
      observer.disconnect();
    };
  }, []);

  return (
    <div className="w-full bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden pt-20 pb-10">
        <div className="container px-4 mx-auto relative z-10">
          <div className="flex flex-col md:flex-row items-center">
            <div className="md:w-1/2 mb-10 md:mb-0">
              <motion.h1
                className="text-4xl md:text-5xl font-bold text-slate-900 dark:text-white mb-6"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                AudioNav: Vision Through Sound
              </motion.h1>
              <motion.p
                className="text-lg text-slate-700 dark:text-slate-300 mb-8 max-w-xl"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                Revolutionary technology that transforms visual information into
                intuitive audio guidance, helping visually impaired users
                navigate the world with confidence and independence.
              </motion.p>
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.4 }}
              >
                <Button className="bg-primary text-slate-50 dark:text-slate-50 px-6 py-3 rounded-lg font-medium shadow-lg hover:shadow-xl hover:bg-primary/90 dark:hover:bg-primary/80 transition-all mr-4">
                  Learn More
                </Button>
                <Button className="border border-primary text-primary dark:text-primary-foreground px-6 py-3 rounded-lg font-medium hover:bg-primary/5 dark:hover:bg-primary/20 transition-all">
                  How It Works
                </Button>
              </motion.div>
            </div>
            <div className="md:w-1/2 relative">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.7 }}
                className="relative w-full h-[400px] rounded-2xl overflow-hidden shadow-2xl"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-primary/20 to-primary/10 z-10 rounded-2xl">
                  <div className="absolute bottom-6 left-6 right-6 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm p-4 rounded-xl z-20">
                    <p className="font-medium text-primary dark:text-primary-foreground">
                      AudioNav is actively identifying:
                    </p>
                    <p className="text-slate-700 dark:text-slate-300">
                      Crosswalk ahead • Person approaching • Store entrance on
                      right
                    </p>
                  </div>
                </div>
                <div className="absolute inset-0 flex items-center justify-center bg-slate-200 dark:bg-slate-700">
                  <MapPin className="h-24 w-24 text-primary/40" />
                </div>
              </motion.div>
            </div>
          </div>
        </div>
        <div className="absolute w-[500px] h-[500px] rounded-full bg-primary/10 dark:bg-primary/5 -top-64 -right-64 blur-3xl"></div>
        <div className="absolute w-[300px] h-[300px] rounded-full bg-blue-500/10 dark:bg-blue-500/5 bottom-10 -left-32 blur-3xl"></div>
      </section>

      {/* How It Works Section */}
      <section className="py-16 relative">
        <div className="container px-4 mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-4">
              How AudioNav Works
            </h2>
            <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              Our groundbreaking technology converts visual information into
              intuitive audio guidance, creating a seamless experience for
              users. Here's the journey from sight to sound:
            </p>
          </div>

          <div className="relative mt-20">
            {/* Connection Line */}
            <div className="absolute left-1/2 top-0 bottom-0 w-1 bg-primary/20 dark:bg-primary/30 -translate-x-1/2 hidden md:block"></div>

            {/* Steps */}
            <div className="space-y-24 relative">
              {steps.map((step, index) => (
                <div
                  key={step.id}
                  id={`step-${step.id}`}
                  className={cn(
                    "observe-me flex flex-col md:flex-row items-center gap-8 relative",
                    index % 2 === 1 ? "md:flex-row-reverse" : ""
                  )}
                >
                  {/* Step number with icon */}
                  <motion.div
                    className={cn(
                      "flex-shrink-0 w-20 h-20 rounded-full flex items-center justify-center z-10 shadow-lg",
                      step.color,
                      step.color === "bg-blue-100" ? "dark:bg-blue-900/50" : "",
                      step.color === "bg-green-100"
                        ? "dark:bg-green-900/50"
                        : "",
                      step.color === "bg-purple-100"
                        ? "dark:bg-purple-900/50"
                        : "",
                      step.color === "bg-orange-100"
                        ? "dark:bg-orange-900/50"
                        : "",
                      step.color === "bg-red-100" ? "dark:bg-red-900/50" : ""
                    )}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={
                      isVisible[`step-${step.id}`]
                        ? { opacity: 1, scale: 1 }
                        : {}
                    }
                    transition={{ duration: 0.5 }}
                  >
                    {step.icon}
                  </motion.div>

                  {/* Content */}
                  <motion.div
                    className="md:w-[calc(50%-2.5rem)] bg-white dark:bg-slate-800 p-6 rounded-xl shadow-md"
                    initial={{ opacity: 0, y: 20 }}
                    animate={
                      isVisible[`step-${step.id}`] ? { opacity: 1, y: 0 } : {}
                    }
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    <h3 className="text-xl font-bold text-slate-900 dark:text-white mb-2 flex items-center">
                      <span className="w-8 h-8 bg-primary text-white rounded-full flex items-center justify-center mr-2 flex-shrink-0">
                        {step.id}
                      </span>
                      {step.title}
                    </h3>
                    <p className="text-slate-700 dark:text-slate-300 mb-4">
                      {step.description}
                    </p>
                    <div className="bg-slate-50 dark:bg-slate-700 p-4 rounded-lg text-sm text-slate-700 dark:text-slate-300">
                      {step.details}
                    </div>
                  </motion.div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Use Cases Tabs */}
      <section className="py-16 bg-white dark:bg-slate-900">
        <div className="container px-4 mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-4">
              AudioNav in Action
            </h2>
            <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              See how AudioNav helps users navigate different environments with
              confidence
            </p>
          </div>

          <Tabs defaultValue="street" className="w-full max-w-4xl mx-auto">
            <TabsList className="grid grid-cols-3 gap-4 mb-8 bg-slate-100 dark:bg-slate-800">
              <TabsTrigger
                value="street"
                className="text-slate-700 dark:text-slate-300 data-[state=active]:bg-primary data-[state=active]:text-white"
              >
                Street Navigation
              </TabsTrigger>
              <TabsTrigger
                value="indoor"
                className="text-slate-700 dark:text-slate-300 data-[state=active]:bg-primary data-[state=active]:text-white"
              >
                Indoor Spaces
              </TabsTrigger>
              <TabsTrigger
                value="public"
                className="text-slate-700 dark:text-slate-300 data-[state=active]:bg-primary data-[state=active]:text-white"
              >
                Public Transit
              </TabsTrigger>
            </TabsList>

            <div className="bg-slate-50 dark:bg-slate-800 p-6 rounded-xl">
              <TabsContent value="street" className="mt-0">
                <div className="flex flex-col md:flex-row gap-6">
                  <div className="md:w-1/2 relative h-[300px] rounded-lg overflow-hidden bg-slate-200 dark:bg-slate-700 flex items-center justify-center">
                    <Camera className="h-16 w-16 text-slate-400 dark:text-slate-500" />
                  </div>
                  <div className="md:w-1/2">
                    <h3 className="text-xl font-bold mb-4 dark:text-white">
                      Street Navigation
                    </h3>
                    <div className="space-y-4">
                      <div className="bg-white dark:bg-slate-700 p-4 rounded-lg border-l-4 border-primary">
                        <p className="italic text-slate-700 dark:text-slate-300">
                          "Crosswalk ahead in 15 feet. Traffic light is red. Two
                          pedestrians waiting."
                        </p>
                      </div>
                      <div className="bg-white dark:bg-slate-700 p-4 rounded-lg border-l-4 border-yellow-500">
                        <p className="italic text-slate-700 dark:text-slate-300">
                          "Construction on your right. Temporary walkway
                          available. Follow the path left of the barrier."
                        </p>
                      </div>
                      <div className="bg-white dark:bg-slate-700 p-4 rounded-lg border-l-4 border-green-500">
                        <p className="italic text-slate-700 dark:text-slate-300">
                          "Your favorite coffee shop 'Morning Brew' is 30 feet
                          ahead on your left. Door is currently open."
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="indoor" className="mt-0">
                <div className="flex flex-col md:flex-row gap-6">
                  <div className="md:w-1/2 relative h-[300px] rounded-lg overflow-hidden bg-slate-200 dark:bg-slate-700 flex items-center justify-center">
                    <Building className="h-16 w-16 text-slate-400 dark:text-slate-500" />
                  </div>
                  <div className="md:w-1/2">
                    <h3 className="text-xl font-bold mb-4 dark:text-white">
                      Indoor Spaces
                    </h3>
                    <div className="space-y-4">
                      <div className="bg-white dark:bg-slate-700 p-4 rounded-lg border-l-4 border-primary">
                        <p className="italic text-slate-700 dark:text-slate-300">
                          "Department store entrance. Clothing section straight
                          ahead. Escalator to your right leads to electronics."
                        </p>
                      </div>
                      <div className="bg-white dark:bg-slate-700 p-4 rounded-lg border-l-4 border-yellow-500">
                        <p className="italic text-slate-700 dark:text-slate-300">
                          "Restaurant seating area. Empty table for four at your
                          2 o'clock, approximately 10 feet away."
                        </p>
                      </div>
                      <div className="bg-white dark:bg-slate-700 p-4 rounded-lg border-l-4 border-green-500">
                        <p className="italic text-slate-700 dark:text-slate-300">
                          "Office lobby. Reception desk straight ahead. Elevator
                          bank to your left. Restrooms located past reception on
                          the right."
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="public" className="mt-0">
                <div className="flex flex-col md:flex-row gap-6">
                  <div className="md:w-1/2 relative h-[300px] rounded-lg overflow-hidden bg-slate-200 dark:bg-slate-700 flex items-center justify-center">
                    <Bus className="h-16 w-16 text-slate-400 dark:text-slate-500" />
                  </div>
                  <div className="md:w-1/2">
                    <h3 className="text-xl font-bold mb-4 dark:text-white">
                      Public Transit
                    </h3>
                    <div className="space-y-4">
                      <div className="bg-white dark:bg-slate-700 p-4 rounded-lg border-l-4 border-primary">
                        <p className="italic text-slate-700 dark:text-slate-300">
                          "Bus 42 approaching. This is your route to downtown.
                          Bus will arrive in approximately 2 minutes."
                        </p>
                      </div>
                      <div className="bg-white dark:bg-slate-700 p-4 rounded-lg border-l-4 border-yellow-500">
                        <p className="italic text-slate-700 dark:text-slate-300">
                          "Subway platform. Train approaching from left. Stand
                          behind the yellow tactile strip. Doors will open in
                          front of you."
                        </p>
                      </div>
                      <div className="bg-white dark:bg-slate-700 p-4 rounded-lg border-l-4 border-green-500">
                        <p className="italic text-slate-700 dark:text-slate-300">
                          "You are in train car 5. Your stop 'Central Station'
                          will be the next stop in approximately 3 minutes. Exit
                          doors will be on your right."
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </div>
          </Tabs>
        </div>
      </section>

      {/* Key Features */}
      <section className="py-16 bg-gradient-to-b from-white to-slate-50 dark:from-slate-900 dark:to-slate-800">
        <div className="container px-4 mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-4">
              Key Features
            </h2>
            <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
              AudioNav goes beyond basic navigation to provide a comprehensive
              solution
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                className="bg-white dark:bg-slate-800 p-6 rounded-xl shadow-md"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true, margin: "-50px" }}
              >
                <div className="flex items-start gap-4">
                  <div className="p-3 rounded-lg bg-slate-100 dark:bg-slate-700">
                    {feature.icon}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                      {feature.title}
                    </h3>
                    <p className="text-slate-600 dark:text-slate-300">
                      {feature.description}
                    </p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ or Technical Details could be added here */}
    </div>
  );
}
