/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Keep Vercel deployment focused on TypeScript/build errors.
  // This prevents style-only ESLint warnings from blocking deployment.
  eslint: {
    ignoreDuringBuilds: true
  },

  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      fs: false,
      path: false,
      crypto: false
    };
    return config;
  }
};

export default nextConfig;
