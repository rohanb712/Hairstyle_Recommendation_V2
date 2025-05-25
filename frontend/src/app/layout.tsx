import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AI Men\'s Hairstyle Recommender & Visualizer',
  description: 'Discover your perfect hairstyle with AI-powered recommendations and virtual try-on for men',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <div className="min-h-screen bg-gradient-to-br from-slate-100 to-gray-200">
          {children}
        </div>
      </body>
    </html>
  )
} 