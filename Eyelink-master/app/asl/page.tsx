"use client";

import React, { useRef, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Camera,
  Mic,
  HandMetal,
  Settings,
  Download,
  Copy,
  RefreshCw,
  MicOff,
  Video,
  VideoOff,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { SparklesText } from "@/components/magicui/sparkles-text";
import { AuroraText } from "@/components/magicui/aurora-text";

interface ProviderPageProps {
  params: {
    provider: string;
  };
}

export default function ASLPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const [isRecording, setIsRecording] = React.useState(false);
  const [isMicEnabled, setIsMicEnabled] = React.useState(false);
  const [isVideoEnabled, setIsVideoEnabled] = React.useState(true);
  const [translatedText, setTranslatedText] = React.useState("");
  const [recordedChunks, setRecordedChunks] = React.useState<Blob[]>([]);
  const [selectedDevices, setSelectedDevices] = React.useState({
    videoId: "",
    audioId: "",
  });
  const [devices, setDevices] = React.useState<{
    video: MediaDeviceInfo[];
    audio: MediaDeviceInfo[];
  }>({
    video: [],
    audio: [],
  });
  const [uploadedVideo, setUploadedVideo] = React.useState<string | null>(null);

  // Get available media devices
  useEffect(() => {
    async function getDevices() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        setDevices({
          video: devices.filter((device) => device.kind === "videoinput"),
          audio: devices.filter((device) => device.kind === "audioinput"),
        });
      } catch (error) {
        console.error("Error getting devices:", error);
        toast.error("Failed to get media devices");
      }
    }

    getDevices();
  }, []);

  // Initialize video stream
  useEffect(() => {
    async function setupStream() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: isVideoEnabled,
          audio: isMicEnabled,
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        // Setup media recorder
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            setRecordedChunks((prev) => [...prev, event.data]);
          }
        };

        mediaRecorder.onstop = () => {
          const blob = new Blob(recordedChunks, {
            type: "video/webm",
          });
          // Here you would typically send the blob to your ASL translation service
          console.log("Recording stopped, blob created:", blob);
          // Simulate translation
          setTranslatedText(
            "This is a simulated translation of the recorded ASL signs..."
          );
        };
      } catch (error) {
        console.error("Error accessing media devices:", error);
        toast.error("Failed to access camera/microphone");
      }
    }

    if (isVideoEnabled || isMicEnabled) {
      setupStream();
    }

    return () => {
      // Cleanup
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, [isVideoEnabled, isMicEnabled]);

  const handleStartRecording = () => {
    setRecordedChunks([]);
    mediaRecorderRef.current?.start();
    setIsRecording(true);
    toast.success("Recording started");
  };

  const handleStopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsRecording(false);

    // Stop all tracks in the stream
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    // Turn off video and mic
    setIsVideoEnabled(false);
    setIsMicEnabled(false);

    toast.success("Recording stopped");
  };

  const handleFileUpload = (
    event:
      | React.ChangeEvent<HTMLInputElement>
      | React.DragEvent<HTMLLabelElement>
  ) => {
    let file: File | null = null;

    if (event.type === "change") {
      file =
        (event as React.ChangeEvent<HTMLInputElement>).target.files?.[0] ||
        null;
    } else if (event.type === "drop") {
      file =
        (event as React.DragEvent<HTMLLabelElement>).dataTransfer?.files[0] ||
        null;
      event.preventDefault();
    }

    if (file && file.type.startsWith("video/")) {
      const videoUrl = URL.createObjectURL(file);
      setUploadedVideo(videoUrl);
      setTranslatedText(`Processing uploaded file: ${file.name}`);

      // Clean up the previous video URL if it exists
      if (uploadedVideo) {
        URL.revokeObjectURL(uploadedVideo);
      }
    } else if (file) {
      toast.error("Please upload a valid video file");
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    event.currentTarget.classList.add("border-primary");
  };

  const handleDragLeave = (event: React.DragEvent) => {
    event.preventDefault();
    event.currentTarget.classList.remove("border-primary");
  };

  // Clean up object URL when component unmounts
  React.useEffect(() => {
    return () => {
      if (uploadedVideo) {
        URL.revokeObjectURL(uploadedVideo);
      }
    };
  }, []);

  const toggleMicrophone = () => {
    setIsMicEnabled(!isMicEnabled);
  };

  const toggleVideo = () => {
    setIsVideoEnabled(!isVideoEnabled);
  };

  const copyTranslation = () => {
    if (translatedText) {
      navigator.clipboard.writeText(translatedText);
      toast.success("Translation copied to clipboard");
    }
  };

  const downloadTranslation = () => {
    if (translatedText) {
      const blob = new Blob([translatedText], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "translation.txt";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success("Translation downloaded");
    }
  };

  return (
    <div className="mt-24">
      {/* Page Header */}
      <div className="relative text-center mb-16 container max-w-7xl mx-auto px-6">
        <div className="space-y-12">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
            ASL Translation
          </h1>
          <p className="text-lg sm:text-xl text-muted-foreground max-w-3xl mx-auto">
            Record or upload ASL videos for real-time translation. Our
            AI-powered system provides accurate and instant translations to
            bridge communication gaps.
          </p>
        </div>
      </div>

      <div className="container max-w-7xl mx-auto px-6 pb-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            <Card className="p-6 shadow-lg">
              <Tabs defaultValue="camera" className="w-full">
                <TabsList className="grid w-full grid-cols-2 mb-6">
                  <TabsTrigger
                    value="camera"
                    className="flex items-center gap-3 py-3"
                  >
                    <Camera className="h-5 w-5" />
                    Camera Input
                  </TabsTrigger>
                  <TabsTrigger
                    value="upload"
                    className="flex items-center gap-3 py-3"
                  >
                    <HandMetal className="h-5 w-5" />
                    Upload Video
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="camera" className="space-y-6">
                  {/* Camera Preview */}
                  <div className="aspect-video bg-muted rounded-xl overflow-hidden relative border border-border/50">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                    />
                    {!isVideoEnabled && (
                      <div className="absolute inset-0 flex items-center justify-center backdrop-blur-sm bg-black/5">
                        <span className="text-muted-foreground font-medium text-sm sm:text-base">
                          Camera is disabled
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Camera Controls */}
                  <div className="flex flex-col sm:flex-row items-center gap-4">
                    <Button
                      size="lg"
                      className="flex-1 h-12 text-base font-medium transition-all w-full sm:w-auto"
                      onClick={
                        isRecording ? handleStopRecording : handleStartRecording
                      }
                      disabled={!isVideoEnabled && !isMicEnabled}
                    >
                      {isRecording ? "Stop" : "Start Recording"}
                    </Button>
                    <div className="flex gap-4 w-full sm:w-auto">
                      <Button
                        size="icon"
                        variant="outline"
                        className="h-12 w-12 flex-shrink-0"
                        onClick={toggleVideo}
                      >
                        {isVideoEnabled ? (
                          <Video className="h-5 w-5" />
                        ) : (
                          <VideoOff className="h-5 w-5" />
                        )}
                      </Button>
                      <Button
                        size="icon"
                        variant="outline"
                        className="h-12 w-12 flex-shrink-0"
                        onClick={toggleMicrophone}
                      >
                        {isMicEnabled ? (
                          <Mic className="h-5 w-5" />
                        ) : (
                          <MicOff className="h-5 w-5" />
                        )}
                      </Button>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="upload">
                  <label
                    htmlFor="video-upload"
                    className="aspect-video bg-muted/50 rounded-xl flex items-center justify-center border-2 border-dashed border-border/50 hover:border-primary/50 transition-colors cursor-pointer relative overflow-hidden"
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleFileUpload}
                  >
                    {uploadedVideo ? (
                      <>
                        <video
                          src={uploadedVideo}
                          className="w-full h-full object-cover"
                          controls
                        />
                        <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                          <p className="text-white font-medium text-sm sm:text-base text-center px-4">
                            Click or drag to replace video
                          </p>
                        </div>
                      </>
                    ) : (
                      <div className="text-center space-y-4 px-4 sm:px-6">
                        <p className="text-muted-foreground text-sm sm:text-base">
                          Drag and drop a video file or click to upload
                        </p>
                        <Button variant="outline" className="h-11 px-4">
                          Choose File
                        </Button>
                        <input
                          id="video-upload"
                          type="file"
                          accept="video/*"
                          className="hidden"
                          onChange={handleFileUpload}
                        />
                      </div>
                    )}
                  </label>
                </TabsContent>
              </Tabs>
            </Card>

            {/* Translation Settings */}
            <Card className="p-6 shadow-lg">
              <h3 className="font-semibold text-lg mb-6">
                Translation Settings
              </h3>
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-3">
                  <label className="text-sm font-medium text-muted-foreground">
                    Output Language
                  </label>
                  <Select defaultValue="en">
                    <SelectTrigger className="h-11">
                      <SelectValue placeholder="Select language" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="en">English</SelectItem>
                      <SelectItem value="es">Spanish</SelectItem>
                      <SelectItem value="fr">French</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-3">
                  <label className="text-sm font-medium text-muted-foreground">
                    Translation Mode
                  </label>
                  <Select defaultValue="realtime">
                    <SelectTrigger className="h-11">
                      <SelectValue placeholder="Select mode" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="realtime">Real-time</SelectItem>
                      <SelectItem value="batch">Batch</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </Card>
          </div>

          {/* Output Section */}
          <div className="space-y-6">
            <Card className="p-6 shadow-lg">
              <div className="flex items-center justify-between mb-6">
                <h3 className="font-semibold text-lg">Translation Output</h3>
                <div className="flex items-center gap-2">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="hover:bg-primary/10"
                    onClick={copyTranslation}
                  >
                    <Copy className="h-5 w-5" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="hover:bg-primary/10"
                    onClick={downloadTranslation}
                  >
                    <Download className="h-5 w-5" />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    className="hover:bg-primary/10"
                    onClick={() => setTranslatedText("")}
                  >
                    <RefreshCw className="h-5 w-5" />
                  </Button>
                </div>
              </div>
              <div className="min-h-[400px] bg-muted/30 rounded-xl p-6 border border-border/50">
                {translatedText ? (
                  <p className="text-base leading-relaxed">{translatedText}</p>
                ) : (
                  <p className="text-muted-foreground text-center mt-12 font-medium">
                    Translation will appear here...
                  </p>
                )}
              </div>
            </Card>

            {/* Translation History */}
            <Card className="p-6 shadow-lg">
              <h3 className="font-semibold text-lg mb-6">
                Recent Translations
              </h3>
              <div className="space-y-2">
                <p className="text-sm text-muted-foreground font-medium">
                  No recent translations
                </p>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
