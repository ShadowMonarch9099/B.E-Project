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
  title: "Sakshya · Ask your documents",
  description: "Citation-grounded answers over English, Hindi and Marathi documents, including scans.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
