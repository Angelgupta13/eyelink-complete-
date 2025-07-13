import React from "react";
import {
  FaEye,
  FaFacebookF,
  FaTwitter,
  FaInstagram,
  FaLinkedinIn,
  FaPhone,
  FaEnvelope,
  FaMapMarkerAlt,
} from "react-icons/fa";
import Link from "next/link";

export default function Footer() {
  return (
    <footer className="w-full bg-gradient-to-b from-background to-background/50 border-t border-border/10">
      <div className="container mx-auto px-4 py-16">
        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
          {/* Brand Section */}
          <div className="space-y-6">
            <Link href="/" className="flex items-center gap-3 group">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg group-hover:shadow-blue-500/25 transition-shadow duration-500">
                <FaEye className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-600">
                  Eyelink
                </h3>
                <p className="text-xs text-muted-foreground">
                  Bridging Hearts, Paving Ways
                </p>
              </div>
            </Link>

            <p className="text-sm text-muted-foreground/80 leading-relaxed">
              Empowering differently-abled individuals through innovative
              solutions for accessible communication, mobility, and dedicated
              care services.
            </p>

            <div className="flex items-center gap-4">
              {[FaFacebookF, FaTwitter, FaInstagram, FaLinkedinIn].map(
                (Icon, i) => (
                  <a
                    key={i}
                    href="#"
                    className="w-8 h-8 rounded-lg bg-muted/50 flex items-center justify-center text-muted-foreground hover:bg-blue-500 hover:text-white transition-all duration-300"
                  >
                    <Icon className="w-4 h-4" />
                  </a>
                )
              )}
            </div>
          </div>

          {/* Quick Links */}
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">Quick Links</h3>
            <ul className="space-y-3">
              {[
                { text: "ASL Translator", href: "/asl" },
                { text: "Specialized Cabs", href: "/cabs" },
                { text: "Care Providers", href: "/care" },
                { text: "About Us", href: "/about" },
                { text: "Contact", href: "/contact" },
              ].map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground/80 hover:text-blue-500 transition-colors duration-200 flex items-center gap-2"
                  >
                    <div className="w-1 h-1 rounded-full bg-blue-500/50" />
                    {link.text}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Services */}
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">Our Services</h3>
            <ul className="space-y-3">
              {[
                { text: "Live ASL Translation", href: "/asl/live" },
                { text: "Text to ASL", href: "/asl/text" },
                { text: "Emergency Transport", href: "/cabs/emergency" },
                { text: "Care Matching", href: "/care/match" },
                { text: "Support & Training", href: "/support" },
              ].map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground/80 hover:text-blue-500 transition-colors duration-200 flex items-center gap-2"
                  >
                    <div className="w-1 h-1 rounded-full bg-blue-500/50" />
                    {link.text}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact Info */}
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">Get in Touch</h3>
            <ul className="space-y-4">
              {[
                {
                  icon: FaMapMarkerAlt,
                  content:
                    "123 Inclusivity Plaza, Accessibility District, MD 54321",
                },
                {
                  icon: FaPhone,
                  content: "1-800-EYE-LINK",
                  href: "tel:1-800-EYE-LINK",
                },
                {
                  icon: FaEnvelope,
                  content: "support@eyelink.com",
                  href: "mailto:support@eyelink.com",
                },
              ].map((item, i) => (
                <li
                  key={i}
                  className="flex gap-3 text-sm text-muted-foreground/80"
                >
                  <item.icon className="w-5 h-5 text-blue-500 shrink-0 mt-0.5" />
                  {item.href ? (
                    <a
                      href={item.href}
                      className="hover:text-blue-500 transition-colors duration-200"
                    >
                      {item.content}
                    </a>
                  ) : (
                    <span>{item.content}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-16 pt-8 border-t border-border/10">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-sm text-muted-foreground/70">
              © {new Date().getFullYear()} Eyelink. All rights reserved.
            </p>
            <div className="flex flex-wrap justify-center gap-8">
              {[
                { text: "Privacy Policy", href: "/privacy" },
                { text: "Terms of Service", href: "/terms" },
                { text: "Accessibility", href: "/accessibility" },
              ].map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="text-sm text-muted-foreground/70 hover:text-blue-500 transition-colors duration-200"
                >
                  {link.text}
                </Link>
              ))}
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
