import type { Metadata } from "next";
import { Hind, IBM_Plex_Mono, Mukta } from "next/font/google";
import "./globals.css";

// The Sakshya artifact's typefaces: Mukta (display) and Hind (body) both cover
// Devanagari; IBM Plex Mono for data.
const display = Mukta({
  subsets: ["latin", "devanagari"],
  weight: ["400", "600", "700", "800"],
  variable: "--font-display",
});
const body = Hind({
  subsets: ["latin", "devanagari"],
  weight: ["400", "500", "600"],
  variable: "--font-body",
});
const mono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-mono" });

export const metadata: Metadata = {
  title: "Sakshya · Document type detection",
  description: "Upload any file and see what type of document it is, detected from its content.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
