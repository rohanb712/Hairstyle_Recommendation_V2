'use client'

import { useState } from 'react'
import { Wand2, Heart } from 'lucide-react'
import { HairstyleResponse } from '@/types/api'

interface HairstyleCardProps {
  hairstyle: HairstyleResponse
  isSelected?: boolean
  onClick?: () => void
}

export default function HairstyleCard({ hairstyle, isSelected = false, onClick }: HairstyleCardProps) {
  const [imageLoaded, setImageLoaded] = useState(false)
  const [imageError, setImageError] = useState(false)

  const handleImageLoad = () => {
    setImageLoaded(true)
  }

  const handleImageError = () => {
    setImageError(true)
    setImageLoaded(true)
  }

  return (
    <div
      className={`rounded-xl border-2 transition-all cursor-pointer ${
        isSelected
          ? 'border-primary-500 bg-primary-50 shadow-lg'
          : 'border-gray-200 hover:border-primary-300 hover:shadow-md'
      }`}
      onClick={onClick}
    >
      <div className="p-6">
        {/* Hairstyle Image */}
        <div className="aspect-w-4 aspect-h-3 mb-4">
          {!imageLoaded && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500" />
            </div>
          )}
          
          {!imageError ? (
            <img
              src={hairstyle.image_url}
              alt={hairstyle.name}
              className={`w-full h-full object-cover rounded-lg ${
                imageLoaded ? 'opacity-100' : 'opacity-0'
              }`}
              onLoad={handleImageLoad}
              onError={handleImageError}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center bg-gray-100">
              <div className="text-center text-gray-500">
                <Heart className="h-12 w-12 mx-auto mb-2" />
                <p className="text-sm">Image not available</p>
              </div>
            </div>
          )}
        </div>

        {/* Hairstyle Info */}
        <div className="space-y-3">
          <h3 className="text-lg font-bold text-gray-800">{hairstyle.name}</h3>
          <p className="text-gray-600 text-sm leading-relaxed">{hairstyle.description}</p>
          
          {/* Face Shape Compatibility */}
          {hairstyle.suitable_face_shapes && hairstyle.suitable_face_shapes.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {hairstyle.suitable_face_shapes.map((shape) => (
                <span
                  key={shape}
                  className="px-3 py-1 bg-primary-100 text-primary-700 text-xs font-medium rounded-full"
                >
                  {shape}
                </span>
              ))}
            </div>
          )}
        </div>

        {/* Selection Indicator */}
        {isSelected && (
          <div className="mt-4 flex items-center justify-center">
            <div className="w-6 h-6 bg-primary-500 rounded-full flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
              </svg>
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 