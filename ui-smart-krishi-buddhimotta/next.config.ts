import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Allow the Arena preview host to load Next.js dev assets and HMR.
  allowedDevOrigins: ["*.e2b.app"],
};

export default nextConfig;
