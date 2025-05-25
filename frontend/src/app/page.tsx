'use client'

import { useState } from 'react'
import ImageUploader from '@/components/ImageUploader'
import FaceAnalysisDisplay from '@/components/FaceAnalysisDisplay'
import HairstyleRecommendations from '@/components/HairstyleRecommendations'
import GeneratedImageView from '@/components/GeneratedImageView'
import { FaceAnalysisResponse, HairstyleResponse } from '@/types/api'

export default function Home() {
  const [step, setStep] = useState<'upload' | 'analysis' | 'recommendations' | 'generation'>('upload')
  const [uploadedImages, setUploadedImages] = useState<File[]>([])
  const [faceAnalysis, setFaceAnalysis] = useState<FaceAnalysisResponse | null>(null)
  const [selectedHairstyle, setSelectedHairstyle] = useState<HairstyleResponse | null>(null)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

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
    setStep('upload')
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
            AI Men's Hairstyle Recommender
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            Discover your perfect men's hairstyle with AI-powered recommendations and virtual try-on
          </p>
          
          {/* Progress Indicator */}
          <div className="flex justify-center items-center space-x-4 mb-8">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'upload' ? 'bg-primary-600 text-white' : 
              ['analysis', 'recommendations', 'generation'].includes(step) ? 'bg-primary-200 text-primary-800' : 
              'bg-gray-200 text-gray-500'
            }`}>
              1
            </div>
            <div className={`w-16 h-1 ${
              ['analysis', 'recommendations', 'generation'].includes(step) ? 'bg-primary-300' : 'bg-gray-200'
            }`} />
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'analysis' ? 'bg-primary-600 text-white' : 
              ['recommendations', 'generation'].includes(step) ? 'bg-primary-200 text-primary-800' : 
              'bg-gray-200 text-gray-500'
            }`}>
              2
            </div>
            <div className={`w-16 h-1 ${
              ['recommendations', 'generation'].includes(step) ? 'bg-primary-300' : 'bg-gray-200'
            }`} />
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'recommendations' ? 'bg-primary-600 text-white' : 
              step === 'generation' ? 'bg-primary-200 text-primary-800' : 
              'bg-gray-200 text-gray-500'
            }`}>
              3
            </div>
            <div className={`w-16 h-1 ${
              step === 'generation' ? 'bg-primary-300' : 'bg-gray-200'
            }`} />
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'generation' ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-500'
            }`}>
              4
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          {step === 'upload' && (
            <ImageUploader
              onImagesUploaded={handleImagesUploaded}
              onAnalysisComplete={handleAnalysisComplete}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
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
              onHairstyleSelected={handleHairstyleSelected}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
          )}

          {step === 'generation' && selectedHairstyle && (
            <GeneratedImageView
              hairstyle={selectedHairstyle}
              userImages={uploadedImages}
              onImageGenerated={handleImageGenerated}
              generatedImage={generatedImage}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />
          )}
        </div>

        {/* Reset Button */}
        {step !== 'upload' && (
          <div className="text-center mt-8">
            <button
              onClick={resetWorkflow}
              className="px-6 py-3 bg-accent-600 text-white rounded-lg hover:bg-accent-700 transition-colors"
            >
              Start Over
            </button>
          </div>
        )}
      </div>
    </main>
  )
} 