"use client";

import { Button } from "@/components/ui/button";
import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
} from "@/components/ui/navigation-menu";
import { Menu, MoveRight, X } from "lucide-react";
import { useState } from "react";
import Link from "next/link";
import { ThemeToggle } from "@/components/theme-toggle";
import { useRouter } from "next/navigation";
import { FaEye } from "react-icons/fa";

// Define a type for Navigation Item for better type safety and readability
type NavItem = {
  title: string;
  href: string; // href is required now in the type
  description?: string; // Optional description for dropdown items
  items?: { title: string; href: string }[]; // Optional sub-items for dropdowns
};

export default function Header() {
  const [isOpen, setIsOpen] = useState(false);
  const router = useRouter();

  const navigationItems: NavItem[] = [
    // Use the NavItem type
    {
      title: "ASL Translator",
      href: "/asl",
      description: "Translate sign language to text/speech and vice versa",
    },
    {
      title: "Book Assistant",
      href: "/cabs",
      description:
        "Book accessible transportation for differently-abled people",
    },
    // {
    //   title: "Travel Assistants",
    //   href: "/care",
    //   description: "Find and connect with dedicated care personnel",
    // },
    {
      title: "Audio Navigation",
      href: "/audioNav",
      description: "Auto Navigate your way",
    },
    {
      title: "Wheelchair Accessibility",
      href: "/wheelchair",
      description: "Find wheelchair-friendly locations",
    },
    {
      title: "More", // Group less primary links under "More" for cleaner top-level nav
      href: "#", // Added href for "More" - using '#' as a placeholder as it's a dropdown trigger
      items: [
        {title: "Travel Assistants", href: "/care" },
        { title: "Pricing", href: "/pricing" },
        { title: "FAQ", href: "/faq" },
        { title: "About Us", href: "/about" },
        { title: "Contact", href: "/contact" },
         
      ],
    },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 z-50 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto px-4">
        <nav className="flex h-16 items-center justify-between">
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg">
              <FaEye className="w-7 h-7 text-white" />
            </div>
            <span className="text-xl font-bold">Eyelink</span>
          </Link>

          <div className="hidden md:flex md:items-center md:space-x-6">
            <NavigationMenu>
              <NavigationMenuList>
                {navigationItems.map((item) => (
                  <NavigationMenuItem key={item.title}>
                    {item.items ? (
                      <>
                        <NavigationMenuTrigger>
                          {item.title}
                        </NavigationMenuTrigger>
                        <NavigationMenuContent>
                          <ul className="grid w-[400px] gap-3 p-4 md:w-[500px] md:grid-cols-2">
                            {item.items.map((subItem) => (
                              <li key={subItem.title}>
                                <NavigationMenuLink asChild>
                                  <Link
                                    href={subItem.href}
                                    className="block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                                  >
                                    <div className="text-sm font-medium leading-none">
                                      {subItem.title}
                                    </div>
                                  </Link>
                                </NavigationMenuLink>
                              </li>
                            ))}
                          </ul>
                        </NavigationMenuContent>
                      </>
                    ) : (
                      <NavigationMenuItem>
                        <NavigationMenuLink asChild>
                          <Link
                            href={item.href}
                            className="group inline-flex h-10 w-max items-center justify-center rounded-md bg-background px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[active]:bg-accent/50 data-[state=open]:bg-accent/50"
                          >
                            {item.title}
                          </Link>
                        </NavigationMenuLink>
                      </NavigationMenuItem>
                    )}
                  </NavigationMenuItem>
                ))}
              </NavigationMenuList>
            </NavigationMenu>

            <div className="flex items-center space-x-4">
              <ThemeToggle />
              <Button
                variant="ghost"
                onClick={() => router.push("/sign-in")}
              >
                Login
              </Button>
              {/* <Button onClick={() => router.push("/asl")}>
                Try ASL Translator
                <MoveRight className="ml-2 h-4 w-4" />
              </Button> */}
            </div>
          </div>

          <div className="md:hidden flex items-center space-x-4">
            <ThemeToggle />
            <button onClick={() => setIsOpen(!isOpen)}>
              {isOpen ? (
                <X className="h-6 w-6" />
              ) : (
                <Menu className="h-6 w-6" />
              )}
            </button>
          </div>
        </nav>

        {isOpen && (
          <div className="md:hidden">
            <div className="space-y-4 px-4 pb-4 pt-2">
              {navigationItems.map((item) => (
                <div key={item.title} className="space-y-2">
                  {item.items ? (
                    <>
                      <div className="font-medium">{item.title}</div>
                      <div className="ml-4 space-y-2">
                        {item.items.map((subItem) => (
                          <Link
                            key={subItem.title}
                            href={subItem.href}
                            className="block text-sm text-muted-foreground hover:text-foreground"
                            onClick={() => setIsOpen(false)}
                          >
                            {subItem.title}
                          </Link>
                        ))}
                      </div>
                    </>
                  ) : (
                    <Link
                      href={item.href}
                      className="block font-medium hover:text-foreground"
                      onClick={() => setIsOpen(false)}
                    >
                      {item.title}
                    </Link>
                  )}
                </div>
              ))}
              <div className="space-y-2 border-t pt-4">
                <Button
                  variant="ghost"
                  className="w-full justify-start"
                  onClick={() => {
                    router.push("/sign-in");
                    setIsOpen(false);
                  }}
                >
                  Login
                </Button>
                {/* <Button
                  className="w-full justify-start"
                  onClick={() => {
                    router.push("/asl");
                    setIsOpen(false);
                  }}
                >
                  Try ASL Translator
                  <MoveRight className="ml-2 h-4 w-4" />
                </Button> */}
              </div>
            </div>
          </div>
        )}
      </div>
    </header>
  );
}
