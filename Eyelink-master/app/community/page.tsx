"use client";

import React from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { SparklesText } from "@/components/magicui/sparkles-text";
import {
  Calendar,
  Users,
  MessageCircle,
  Heart,
  MapPin,
  Clock,
  Share2,
  ThumbsUp,
  BookOpen,
  Video,
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

// Mock data for the community page
const upcomingEvents = [
  {
    id: 1,
    title: "ASL Workshop: Beginner's Guide",
    date: "2024-04-15T14:00:00",
    location: "Community Center, Downtown",
    type: "Workshop",
    attendees: 24,
    image: "/events/workshop.jpg",
  },
  {
    id: 2,
    title: "Inclusive Art Exhibition",
    date: "2024-04-20T11:00:00",
    location: "City Art Gallery",
    type: "Exhibition",
    attendees: 45,
    image: "/events/art.jpg",
  },
  {
    id: 3,
    title: "Support Group Meeting",
    date: "2024-04-18T18:30:00",
    location: "Virtual Meeting",
    type: "Support",
    attendees: 15,
    image: "/events/support.jpg",
  },
];

const supportGroups = [
  {
    id: 1,
    name: "Parents Support Network",
    members: 156,
    description:
      "A supportive community for parents of differently-abled children.",
    meetingTime: "Every Tuesday, 7 PM",
    type: "Virtual & In-person",
  },
  {
    id: 2,
    name: "ASL Learning Circle",
    members: 89,
    description:
      "Practice ASL with others in a friendly, supportive environment.",
    meetingTime: "Weekends, 11 AM",
    type: "In-person",
  },
  {
    id: 3,
    name: "Caregivers Connect",
    members: 234,
    description: "Share experiences and support with fellow caregivers.",
    meetingTime: "Wednesday & Sunday, 6 PM",
    type: "Virtual",
  },
];

const stories = [
  {
    id: 1,
    author: {
      name: "Sarah Johnson",
      image: "/avatars/sarah.jpg",
    },
    title: "Finding My Voice Through ASL",
    preview: "My journey of learning ASL changed not just how I communicate...",
    likes: 45,
    comments: 12,
    tags: ["Inspiration", "ASL", "Personal Story"],
  },
  {
    id: 2,
    author: {
      name: "Michael Chen",
      image: "/avatars/michael.jpg",
    },
    title: "Breaking Barriers in the Workplace",
    preview: "How inclusive practices transformed our company culture...",
    likes: 67,
    comments: 23,
    tags: ["Workplace", "Inclusion", "Success"],
  },
];

export default function CommunityPage() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20 pt-24">
      {/* Hero Section */}
      <div className="relative text-center mb-16 container max-w-7xl mx-auto px-6">
        <div className="space-y-8">
          <SparklesText
            className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl"
            text="Join Our Community"
          />
          <p className="text-lg sm:text-xl text-muted-foreground max-w-3xl mx-auto">
            Connect, share, and grow with a supportive community dedicated to
            making a difference. Together, we create an inclusive world.
          </p>
          <div className="flex justify-center gap-4">
            <Button size="lg" className="gap-2">
              <Users className="w-5 h-5" />
              Join Community
            </Button>
            <Button size="lg" variant="outline" className="gap-2">
              <Video className="w-5 h-5" />
              Watch Stories
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container max-w-7xl mx-auto px-6 pb-20">
        <Tabs defaultValue="events" className="space-y-8">
          <TabsList className="grid w-full grid-cols-4 max-w-2xl mx-auto">
            <TabsTrigger value="events">Events</TabsTrigger>
            <TabsTrigger value="groups">Support Groups</TabsTrigger>
            <TabsTrigger value="stories">Stories</TabsTrigger>
            <TabsTrigger value="resources">Resources</TabsTrigger>
          </TabsList>

          {/* Events Tab */}
          <TabsContent value="events" className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {upcomingEvents.map((event) => (
                <Card
                  key={event.id}
                  className="p-6 hover:shadow-lg transition-shadow"
                >
                  <div className="space-y-4">
                    <div className="aspect-video relative rounded-lg bg-muted overflow-hidden">
                      <div className="absolute inset-0 bg-gradient-to-br from-primary/20 to-primary/5" />
                      <Badge className="absolute top-2 right-2">
                        {event.type}
                      </Badge>
                    </div>
                    <div>
                      <h3 className="text-xl font-semibold mb-2">
                        {event.title}
                      </h3>
                      <div className="space-y-2 text-sm text-muted-foreground">
                        <div className="flex items-center gap-2">
                          <Calendar className="w-4 h-4" />
                          <span>
                            {new Date(event.date).toLocaleDateString("en-US", {
                              weekday: "long",
                              month: "long",
                              day: "numeric",
                              hour: "numeric",
                              minute: "numeric",
                            })}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <MapPin className="w-4 h-4" />
                          <span>{event.location}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Users className="w-4 h-4" />
                          <span>{event.attendees} attending</span>
                        </div>
                      </div>
                    </div>
                    <Button className="w-full">Join Event</Button>
                  </div>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Support Groups Tab */}
          <TabsContent value="groups" className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {supportGroups.map((group) => (
                <Card
                  key={group.id}
                  className="p-6 hover:shadow-lg transition-shadow"
                >
                  <div className="space-y-4">
                    <div className="flex justify-between items-start">
                      <div>
                        <h3 className="text-xl font-semibold mb-2">
                          {group.name}
                        </h3>
                        <Badge variant="secondary">{group.type}</Badge>
                      </div>
                      <Badge variant="outline" className="flex gap-1">
                        <Users className="w-4 h-4" />
                        {group.members}
                      </Badge>
                    </div>
                    <p className="text-muted-foreground">{group.description}</p>
                    <div className="flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock className="w-4 h-4" />
                      <span>{group.meetingTime}</span>
                    </div>
                    <div className="flex gap-2">
                      <Button className="flex-1">Join Group</Button>
                      <Button variant="outline" size="icon">
                        <Share2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Stories Tab */}
          <TabsContent value="stories" className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {stories.map((story) => (
                <Card
                  key={story.id}
                  className="p-6 hover:shadow-lg transition-shadow"
                >
                  <div className="space-y-4">
                    <div className="flex items-start gap-4">
                      <Avatar>
                        <AvatarImage
                          src={story.author.image}
                          alt={story.author.name}
                        />
                        <AvatarFallback>
                          {story.author.name.slice(0, 2)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex-1">
                        <h3 className="text-xl font-semibold mb-1">
                          {story.title}
                        </h3>
                        <p className="text-sm text-muted-foreground">
                          By {story.author.name}
                        </p>
                      </div>
                    </div>
                    <p className="text-muted-foreground">{story.preview}</p>
                    <div className="flex flex-wrap gap-2">
                      {story.tags.map((tag) => (
                        <Badge
                          key={tag}
                          variant="secondary"
                          className="text-xs"
                        >
                          {tag}
                        </Badge>
                      ))}
                    </div>
                    <div className="flex items-center justify-between pt-4 border-t">
                      <div className="flex gap-4">
                        <Button variant="ghost" size="sm" className="gap-1">
                          <ThumbsUp className="w-4 h-4" />
                          {story.likes}
                        </Button>
                        <Button variant="ghost" size="sm" className="gap-1">
                          <MessageCircle className="w-4 h-4" />
                          {story.comments}
                        </Button>
                      </div>
                      <Button variant="outline" size="sm">
                        Read More
                      </Button>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Resources Tab */}
          <TabsContent value="resources" className="space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <Card className="p-6 hover:shadow-lg transition-shadow">
                <div className="space-y-4">
                  <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <BookOpen className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold">Learning Materials</h3>
                  <p className="text-muted-foreground">
                    Access our curated collection of ASL learning resources,
                    guides, and educational materials.
                  </p>
                  <Button variant="outline" className="w-full">
                    Browse Library
                  </Button>
                </div>
              </Card>

              <Card className="p-6 hover:shadow-lg transition-shadow">
                <div className="space-y-4">
                  <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <Video className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold">Video Tutorials</h3>
                  <p className="text-muted-foreground">
                    Watch expert-led video tutorials on various topics, from
                    basic ASL to advanced communication techniques.
                  </p>
                  <Button variant="outline" className="w-full">
                    Watch Videos
                  </Button>
                </div>
              </Card>

              <Card className="p-6 hover:shadow-lg transition-shadow">
                <div className="space-y-4">
                  <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center">
                    <Heart className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-semibold">Support Resources</h3>
                  <p className="text-muted-foreground">
                    Find information about support services, counseling, and
                    other helpful resources.
                  </p>
                  <Button variant="outline" className="w-full">
                    Get Support
                  </Button>
                </div>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
