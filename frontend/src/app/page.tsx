'use client'

import { useState } from 'react'
import ImageUploader from '@/components/ImageUploader'
import UserProfileForm from '@/components/UserProfileForm'
import FaceAnalysisDisplay from '@/components/FaceAnalysisDisplay'
import HairstyleRecommendations from '@/components/HairstyleRecommendations'
import GeneratedImageView from '@/components/GeneratedImageView'
import { FaceAnalysisResponse, HairstyleResponse, UserProfileResponse } from '@/types/api'

export default function Home() {
  const [step, setStep] = useState<'profile' | 'upload' | 'analysis' | 'recommendations' | 'generation'>('profile')
  const [userProfile, setUserProfile] = useState<UserProfileResponse | null>(null)
  const [uploadedImages, setUploadedImages] = useState<File[]>([])
  const [faceAnalysis, setFaceAnalysis] = useState<FaceAnalysisResponse | null>(null)
  const [selectedHairstyle, setSelectedHairstyle] = useState<HairstyleResponse | null>(null)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleProfileCreated = (profile: UserProfileResponse) => {
    setUserProfile(profile)
    setStep('upload')
  }

  const handleImagesUploaded = (files: File[]) => {
    setUploadedImages(files)
  }

  const handleAnalysisComplete = (analysis: FaceAnalysisResponse) => {
    setFaceAnalysis(analysis)
    setStep('recommendations')
  }

  const handleHairstyleSelected = (hairstyle: HairstyleResponse) => {
    setSelectedHairstyle(hairstyle)
    setStep('generation')
  }

  const handleImageGenerated = (imageUrl: string) => {
    setGeneratedImage(imageUrl)
  }

  const resetWorkflow = () => {
    setStep('profile')
    setUserProfile(null)
    setUploadedImages([])
    setFaceAnalysis(null)
    setSelectedHairstyle(null)
    setGeneratedImage(null)
    setIsLoading(false)
  }

  return (
    <main className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-800 mb-4">
            AI Hairstyle Recommender
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Get personalized hairstyle recommendations with AI-powered face analysis and virtual try-on
          </p>
          
          {/* Progress Indicator */}
          <div className="flex justify-center items-center space-x-2 md:space-x-4 mb-8">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'profile' ? 'bg-blue-600 text-white' : 
              ['upload', 'analysis', 'recommendations', 'generation'].includes(step) ? 'bg-blue-200 text-blue-800' : 
              'bg-gray-200 text-gray-500'
            }`}>
              1
            </div>
            <div className={`w-8 md:w-16 h-1 ${
              ['upload', 'analysis', 'recommendations', 'generation'].includes(step) ? 'bg-blue-300' : 'bg-gray-200'
            }`} />
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'upload' ? 'bg-blue-600 text-white' : 
              ['analysis', 'recommendations', 'generation'].includes(step) ? 'bg-blue-200 text-blue-800' : 
              'bg-gray-200 text-gray-500'
            }`}>
              2
            </div>
            <div className={`w-8 md:w-16 h-1 ${
              ['analysis', 'recommendations', 'generation'].includes(step) ? 'bg-blue-300' : 'bg-gray-200'
            }`} />
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'analysis' ? 'bg-blue-600 text-white' : 
              ['recommendations', 'generation'].includes(step) ? 'bg-blue-200 text-blue-800' : 
              'bg-gray-200 text-gray-500'
            }`}>
              3
            </div>
            <div className={`w-8 md:w-16 h-1 ${
              ['recommendations', 'generation'].includes(step) ? 'bg-blue-300' : 'bg-gray-200'
            }`} />
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'recommendations' ? 'bg-blue-600 text-white' : 
              step === 'generation' ? 'bg-blue-200 text-blue-800' : 
              'bg-gray-200 text-gray-500'
            }`}>
              4
            </div>
            <div className={`w-8 md:w-16 h-1 ${
              step === 'generation' ? 'bg-blue-300' : 'bg-gray-200'
            }`} />
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'generation' ? 'bg-blue-600 text-white' : 'bg-gray-200 text-gray-500'
            }`}>
              5
            </div>
          </div>
          
          {/* Step Labels */}
          <div className="flex justify-center items-center space-x-2 md:space-x-8 text-xs md:text-sm text-gray-600 mb-4">
            <span className={step === 'profile' ? 'text-blue-600 font-medium' : ''}>Profile</span>
            <span className={step === 'upload' ? 'text-blue-600 font-medium' : ''}>Upload</span>
            <span className={step === 'analysis' ? 'text-blue-600 font-medium' : ''}>Analysis</span>
            <span className={step === 'recommendations' ? 'text-blue-600 font-medium' : ''}>Recommendations</span>
            <span className={step === 'generation' ? 'text-blue-600 font-medium' : ''}>Try-On</span>
          </div>
        </div>

        {/* Content */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {step === 'profile' && (
            <UserProfileForm
              onProfileCreated={handleProfileCreated}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
          )}

          {step === 'upload' && userProfile && (
            <div>
              {/* Profile Summary */}
              <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <h3 className="text-lg font-semibold text-blue-800 mb-2">Your Profile</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm text-blue-700">
                  <div><strong>Age:</strong> {userProfile.user_profile.age}</div>
                  <div><strong>Gender:</strong> {userProfile.user_profile.gender}</div>
                  <div><strong>Ethnicity:</strong> {userProfile.user_profile.ethnicity}</div>
                  <div><strong>Hair:</strong> {userProfile.user_profile.hair_texture}</div>
                </div>
              </div>
              
              <ImageUploader
                onImagesUploaded={handleImagesUploaded}
                onAnalysisComplete={handleAnalysisComplete}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
                profileId={userProfile.profile_id}
              />
            </div>
          )}

          {step === 'analysis' && faceAnalysis && (
            <FaceAnalysisDisplay
              analysis={faceAnalysis}
              uploadedImage={uploadedImages[0]}
              onContinue={() => setStep('recommendations')}
            />
          )}

          {step === 'recommendations' && faceAnalysis && (
            <HairstyleRecommendations
              faceShape={faceAnalysis.face_shape}
              userProfile={userProfile?.user_profile}
              onHairstyleSelected={handleHairstyleSelected}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
          )}

          {step === 'generation' && selectedHairstyle && (
            <GeneratedImageView
              hairstyle={selectedHairstyle}
              userImages={uploadedImages}
              userProfile={userProfile?.user_profile}
              onImageGenerated={handleImageGenerated}
              generatedImage={generatedImage}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
          )}
        </div>

        {/* Reset Button */}
        {step !== 'profile' && (
          <div className="text-center mt-8">
            <button
              onClick={resetWorkflow}
              className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Start Over
            </button>
          </div>
        )}
      </div>
    </main>
  )
} 