import type { NextConfig } from "next";

// The browser talks only to Next.js; /api/* is forwarded to the FastAPI backend.
const backend = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  experimental: {
    // Proxied request bodies are cut at 10 MB by default; the backend accepts 50 MB files.
    proxyClientMaxBodySize: "60mb",
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
