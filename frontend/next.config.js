/** @type {import('next').NextConfig} */
const nextConfig = {
    reactStrictMode: true,
    swcMinify: true,
    typescript: {
        // ⚠️ Dangerously allow production builds to successfully complete even if
        // your project has type errors.
        ignoreBuildErrors: true,
    },
    async rewrites() {
        // Only use proxy in development (when NEXT_PUBLIC_API_URL is not set)
        // In production (Vercel), API calls go directly to Render backend
        if (!process.env.NEXT_PUBLIC_API_URL) {
            return [
                {
                    source: '/api/:path*',
                    destination: 'http://127.0.0.1:5000/api/:path*' // Proxy to local backend
                }
            ]
        }
        return []
    }
}

module.exports = nextConfig
