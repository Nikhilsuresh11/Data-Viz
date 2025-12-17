import './globals.css'
import type { Metadata } from 'next'
import Navbar from '@/components/Navbar'

export const metadata: Metadata = {
    title: 'Data-Viz - AI-Powered Data Analysis',
    description: 'Upload, analyze, and visualize your data with AI',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body className="font-sans bg-gray-50 text-gray-900 min-h-screen flex flex-col">
                <Navbar />
                <main className="flex-grow">
                    {children}
                </main>
                <footer className="bg-dark-bg text-white py-8 mt-12">
                    <div className="container mx-auto px-4 text-center text-gray-400">
                        <p>© 2025 Data Viz. Used only Free Tier and open source.</p>
                    </div>
                </footer>
            </body>
        </html>
    )
}
