'use client'

import { useState, useEffect } from 'react'
import { apiService } from '@/services/api'
import { HairstyleResponse, ImageGenerationResponse } from '@/types/api'

interface GeneratedImageViewProps {
  hairstyle: HairstyleResponse
  userImages: File[]
  onImageGenerated: (imageUrl: string) => void
  generatedImage: string | null
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export default function GeneratedImageView({
  hairstyle,
  userImages,
  onImageGenerated,
  generatedImage,
  isLoading,
  setIsLoading,
}: GeneratedImageViewProps) {
  const [error, setError] = useState<string | null>(null)
  const [hasGenerated, setHasGenerated] = useState(false)

  const generateImage = async () => {
    if (!userImages.length || !hairstyle) return

    setIsLoading(true)
    setError(null)

    try {
      // Convert images to base64
      const imagePromises = userImages.map((file) => {
        return new Promise<string>((resolve) => {
          const reader = new FileReader()
          reader.onload = (e) => resolve(e.target?.result as string)
          reader.readAsDataURL(file)
        })
      })

      const base64Images = await Promise.all(imagePromises)

      const response = await apiService.generateHairstyleImage({
        user_images: base64Images,
        hairstyle_id: hairstyle.id,
      })

      onImageGenerated(response.image_url)
      setHasGenerated(true)
    } catch (err: any) {
      console.error('Image generation failed:', err)
      setError('Failed to generate image. Please try again.')
      
      // Fallback: create a placeholder image with text
      const canvas = document.createElement('canvas')
      canvas.width = 400
      canvas.height = 400
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.fillStyle = '#f3f4f6'
        ctx.fillRect(0, 0, 400, 400)
        ctx.fillStyle = '#4b5563'
        ctx.font = '20px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('AI Generation Preview', 200, 180)
        ctx.fillText(`${hairstyle.name}`, 200, 220)
        ctx.font = '14px sans-serif'
        ctx.fillText('(Demo mode - no API key)', 200, 260)
        
        const fallbackUrl = canvas.toDataURL()
        onImageGenerated(fallbackUrl)
        setHasGenerated(true)
      }
    } finally {
      setIsLoading(false)
    }
  }

  const downloadImage = () => {
    if (!generatedImage) return

    const link = document.createElement('a')
    link.href = generatedImage
    link.download = `${hairstyle.name.replace(/\s+/g, '_')}_hairstyle.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const shareImage = async () => {
    if (!generatedImage || !navigator.share) return

    try {
      const response = await fetch(generatedImage)
      const blob = await response.blob()
      const file = new File([blob], `${hairstyle.name}_hairstyle.png`, { type: 'image/png' })

      await navigator.share({
        title: 'My New Hairstyle',
        text: `Check out how I look with ${hairstyle.name}!`,
        files: [file],
      })
    } catch (error) {
      console.error('Sharing failed:', error)
    }
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Virtual Men's Hairstyle Try-On</h2>
        <p className="text-gray-600">
          See how you'd look with <span className="font-semibold text-primary-700">{hairstyle.name}</span>
        </p>
      </div>

      {/* Selected Hairstyle Info */}
      <div className="bg-primary-50 rounded-xl p-6 border border-primary-200">
        <div className="flex items-center space-x-4">
          <img
            src={hairstyle.image_url}
            alt={hairstyle.name}
            className="w-20 h-20 object-cover rounded-lg"
            onError={(e) => {
              const target = e.target as HTMLImageElement
              target.src = 'https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400&h=300&fit=crop&crop=face'
            }}
          />
          <div className="flex-1">
            <h3 className="text-lg font-bold text-gray-800">{hairstyle.name}</h3>
            <p className="text-gray-600 text-sm">{hairstyle.description}</p>
            <div className="flex flex-wrap gap-2 mt-2">
              {hairstyle.suitable_face_shapes.map((shape) => (
                <span
                  key={shape}
                  className="px-2 py-1 bg-primary-200 text-primary-800 text-xs font-medium rounded-full"
                >
                  {shape}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Generation Section */}
      <div className="text-center space-y-6">
        {!hasGenerated && !isLoading && (
          <button
            onClick={generateImage}
            className="px-8 py-4 bg-primary-600 text-white font-medium rounded-xl hover:bg-primary-700 transition-colors flex items-center space-x-3 mx-auto text-lg"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span>Generate AI Preview</span>
          </button>
        )}

        {isLoading && (
          <div className="space-y-4">
            <div className="w-12 h-12 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin mx-auto"></div>
            <div className="space-y-2">
              <p className="text-lg font-medium text-gray-800">Creating Your New Look</p>
              <p className="text-gray-600">AI is processing your photo with the selected men's hairstyle...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-700">{error}</p>
            <button
              onClick={generateImage}
              className="mt-3 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Try Again
            </button>
          </div>
        )}

        {generatedImage && (
          <div className="space-y-6">
            <div className="bg-white rounded-xl shadow-lg p-6 max-w-md mx-auto">
              <img
                src={generatedImage}
                alt={`You with ${hairstyle.name}`}
                className="w-full rounded-lg shadow-md"
              />
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={downloadImage}
                className="px-6 py-3 bg-accent-600 text-white rounded-lg hover:bg-accent-700 transition-colors flex items-center space-x-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                <span>Download</span>
              </button>

              {navigator.share && typeof navigator.share === 'function' && (
                <button
                  onClick={shareImage}
                  className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center space-x-2"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
                  </svg>
                  <span>Share</span>
                </button>
              )}

              <button
                onClick={() => {
                  setHasGenerated(false)
                  onImageGenerated('')
                }}
                className="px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center space-x-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                <span>Try Different Style</span>
              </button>
            </div>

            {/* Results Description */}
            <div className="bg-primary-50 border border-primary-200 rounded-lg p-4 max-w-2xl mx-auto">
              <h4 className="font-semibold text-primary-800 mb-2">How does it look?</h4>
              <p className="text-primary-700 text-sm">
                This AI-generated preview shows how <strong>{hairstyle.name}</strong> would complement your facial features. 
                The style is particularly suited for {hairstyle.suitable_face_shapes.join(', ').toLowerCase()} face shapes 
                and offers a modern, masculine appearance.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 