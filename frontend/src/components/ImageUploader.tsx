'use client'

import React, { useState, useRef } from 'react'
import { apiService } from '@/services/api'
import { FaceAnalysisResponse } from '@/types/api'

interface ImageUploaderProps {
  onImagesUploaded: (files: File[]) => void
  onAnalysisComplete: (analysis: FaceAnalysisResponse) => void
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
}

export default function ImageUploader({ onImagesUploaded, onAnalysisComplete, isLoading, setIsLoading }: ImageUploaderProps) {
  const [dragOver, setDragOver] = useState(false)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    
    const files = Array.from(e.dataTransfer.files).filter(file => 
      file.type.startsWith('image/')
    )
    
    if (files.length > 0) {
      handleFiles(files)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length > 0) {
      handleFiles(files)
    }
  }

  const handleFiles = (files: File[]) => {
    setUploadedFiles(files)
    onImagesUploaded(files)
  }

  const analyzeImages = async () => {
    if (uploadedFiles.length === 0) return

    setIsLoading(true)
    try {
      const analysis = await apiService.analyzeFace(uploadedFiles)
      onAnalysisComplete(analysis)
    } catch (error) {
      console.error('Analysis failed:', error)
      alert('Face analysis failed. Please try with a different image.')
    } finally {
      setIsLoading(false)
    }
  }

  const removeFile = (index: number) => {
    const newFiles = uploadedFiles.filter((_, i) => i !== index)
    setUploadedFiles(newFiles)
    onImagesUploaded(newFiles)
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Upload Your Photo</h2>
        <p className="text-gray-600">Upload a clear front-facing photo to get personalized men's hairstyle recommendations</p>
      </div>

      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${
          dragOver 
            ? 'border-primary-500 bg-primary-50' 
            : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <div className="space-y-4">
          <div className="w-16 h-16 mx-auto bg-primary-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
          </div>
          <div>
            <p className="text-lg font-medium text-gray-700">
              Drop your image here, or click to browse
            </p>
            <p className="text-sm text-gray-500 mt-1">
              Supports JPG, PNG, WebP (max 10MB)
            </p>
          </div>
        </div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileSelect}
        className="hidden"
        multiple
      />

      {/* Uploaded Files Preview */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800">Uploaded Images</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {uploadedFiles.map((file, index) => (
              <div key={index} className="relative group">
                <img
                  src={URL.createObjectURL(file)}
                  alt={`Upload ${index + 1}`}
                  className="w-full h-32 object-cover rounded-lg border border-gray-200"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFile(index)
                  }}
                  className="absolute top-2 right-2 w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center text-xs opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  ×
                </button>
                <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-75 text-white text-xs p-2 rounded-b-lg">
                  {file.name}
                </div>
              </div>
            ))}
          </div>

          {/* Analyze Button */}
          <div className="flex justify-center">
            <button
              onClick={analyzeImages}
              disabled={isLoading}
              className="px-8 py-3 bg-primary-600 text-white font-medium rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                  <span>Analyzing Face...</span>
                </div>
              ) : (
                'Analyze My Face'
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  )
} 