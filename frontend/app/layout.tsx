import './globals.css'
import type { Metadata } from 'next'

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
            <body>{children}</body>
        </html>
    )
}
