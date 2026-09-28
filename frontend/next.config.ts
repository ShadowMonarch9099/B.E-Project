import type { NextConfig } from "next";

// The browser talks only to Next.js; /api/* is forwarded to the FastAPI backend.
const backend = process.env.BACKEND_URL ?? "http://localhost:8000";
// Same setting as the backend's upload limit; the extra 10 MB covers the multipart wrapping.
const maxFileMb = Number(process.env.SAKSHYA_MAX_FILE_MB ?? "200");

const nextConfig: NextConfig = {
  experimental: {
    // Proxied request bodies are cut at 10 MB by default.
    proxyClientMaxBodySize: `${maxFileMb + 10}mb`,
    // OCR of a multi-page scan can take longer than the default 30 s proxy timeout.
    proxyTimeout: 300_000,
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
