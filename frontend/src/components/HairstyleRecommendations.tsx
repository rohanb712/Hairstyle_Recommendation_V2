'use client'

import { useState, useEffect } from 'react'
import { apiService } from '@/services/api'
import { HairstyleResponse, UserProfile } from '@/types/api'
import HairstyleCard from '@/components/HairstyleCard'

interface HairstyleRecommendationsProps {
  faceShape: string
  userProfile?: UserProfile
  onHairstyleSelected: (hairstyle: HairstyleResponse) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export default function HairstyleRecommendations({
  faceShape,
  userProfile,
  onHairstyleSelected,
  isLoading,
  setIsLoading,
}: HairstyleRecommendationsProps) {
  const [hairstyles, setHairstyles] = useState<HairstyleResponse[]>([])
  const [selectedHairstyle, setSelectedHairstyle] = useState<HairstyleResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchRecommendations()
  }, [faceShape, userProfile])

  const fetchRecommendations = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const recommendations = await apiService.getHairstyleRecommendations({
        face_shape: faceShape,
        user_profile: userProfile,
      })
      setHairstyles(recommendations)
    } catch (err: any) {
      setError('Failed to load hairstyle recommendations')
      console.error('Error fetching recommendations:', err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleHairstyleSelect = (hairstyle: HairstyleResponse) => {
    setSelectedHairstyle(hairstyle)
  }

  const handleContinue = () => {
    if (selectedHairstyle) {
      onHairstyleSelected(selectedHairstyle)
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-2">Finding Perfect Men's Styles</h2>
          <p className="text-gray-600">Analyzing your face shape and selecting the best men's hairstyles...</p>
        </div>
        <div className="flex justify-center">
          <div className="w-8 h-8 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin"></div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center space-y-4">
        <h2 className="text-2xl font-bold text-gray-800">Something Went Wrong</h2>
        <p className="text-red-600">{error}</p>
        <button
          onClick={fetchRecommendations}
          className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          Try Again
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Personalized Hairstyle Recommendations</h2>
        <p className="text-gray-600">
          Based on your <span className="font-semibold text-blue-700">{faceShape}</span> face shape
          {userProfile && (
            <>
              , <span className="font-semibold text-blue-700">{userProfile.hair_texture}</span> hair texture
              , and your profile preferences
            </>
          )}
          , here are the best hairstyles for you:
        </p>
      </div>

      {/* Hairstyle Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {hairstyles.map((hairstyle) => (
          <HairstyleCard
            key={hairstyle.id}
            hairstyle={hairstyle}
            isSelected={selectedHairstyle?.id === hairstyle.id}
            onClick={() => handleHairstyleSelect(hairstyle)}
          />
        ))}
      </div>

      {/* Continue Button */}
      {selectedHairstyle && (
        <div className="flex justify-center pt-6">
          <button
            onClick={handleContinue}
            className="px-8 py-3 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 transition-colors flex items-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 4V2a1 1 0 011-1h8a1 1 0 011 1v2m0 0V1a1 1 0 011-1h2a1 1 0 011 1v16.93M7 4H5a1 1 0 00-1 1v16.93M7 4h10M5 21.93h14M12 8v8m4-4H8" />
            </svg>
            <span>Try This Style</span>
          </button>
        </div>
      )}

      {/* Help Text */}
      {!selectedHairstyle && hairstyles.length > 0 && (
        <div className="text-center">
          <p className="text-gray-500 text-sm">
            Select a hairstyle to see how it would look on you with AI virtual try-on
          </p>
        </div>
      )}
    </div>
  )
} 