import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "স্মার্ট কৃষি বুদ্ধিমত্তা | Smart Krishi Buddhimotta",
  description: "BNMPC IT Club-এর স্মার্ট কৃষি বুদ্ধিমত্তা: বাংলাদেশের কৃষকদের জন্য আবহাওয়া, ফসল পরিচর্যা ও AI কৃষি সহায়তা।",
  icons: {
    icon: "/brand-logo.png",
    shortcut: "/brand-logo.png",
    apple: "/brand-logo.png",
  },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="bn">
      <body>{children}</body>
    </html>
  );
}
