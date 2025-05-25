'use client'

import { useRef, useEffect } from 'react'
import { FaceAnalysisResponse } from '@/types/api'

interface FaceAnalysisDisplayProps {
  analysis: FaceAnalysisResponse
  uploadedImage: File
  onContinue: () => void
}

export default function FaceAnalysisDisplay({ analysis, uploadedImage, onContinue }: FaceAnalysisDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (canvasRef.current && uploadedImage) {
      drawAnalysis()
    }
  }, [analysis, uploadedImage])

  const drawAnalysis = () => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return

    const img = new Image()
    img.onload = () => {
      // Set canvas dimensions
      const maxWidth = 500
      const maxHeight = 500
      const ratio = Math.min(maxWidth / img.width, maxHeight / img.height)
      canvas.width = img.width * ratio
      canvas.height = img.height * ratio

      // Draw image
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

      // Draw face landmarks
      if (analysis.landmarks && analysis.landmarks.length > 0) {
        ctx.fillStyle = '#3b82f6'
        ctx.strokeStyle = '#1d4ed8'
        ctx.lineWidth = 2

        analysis.landmarks.forEach((point) => {
          const x = point[0] * ratio
          const y = point[1] * ratio
          
          ctx.beginPath()
          ctx.arc(x, y, 3, 0, 2 * Math.PI)
          ctx.fill()
        })

        // Draw face outline
        if (analysis.landmarks.length >= 17) {
          ctx.beginPath()
          ctx.strokeStyle = '#1d4ed8'
          ctx.lineWidth = 2
          
          // Connect jawline points (typically first 17 landmarks)
          for (let i = 0; i < Math.min(17, analysis.landmarks.length); i++) {
            const x = analysis.landmarks[i][0] * ratio
            const y = analysis.landmarks[i][1] * ratio
            
            if (i === 0) {
              ctx.moveTo(x, y)
            } else {
              ctx.lineTo(x, y)
            }
          }
          ctx.stroke()
        }
      }
    }

    img.src = URL.createObjectURL(uploadedImage)
  }

  const getFaceShapeDescription = (shape: string) => {
    const descriptions: Record<string, string> = {
      'Oval': 'Well-balanced proportions with gentle curves. Most men\'s hairstyles will complement this versatile face shape.',
      'Round': 'Soft features with similar width and height. Angular men\'s cuts and volume on top work best to add definition.',
      'Square': 'Strong jawline with broad forehead. Softer, textured men\'s styles help balance the angular features.',
      'Heart': 'Wider forehead tapering to a narrower chin. Men\'s styles with volume at the jawline create better proportion.',
      'Long': 'Elongated features with high forehead. Shorter men\'s cuts with width on the sides help balance the length.',
      'Diamond': 'Narrow forehead and chin with wider cheekbones. Men\'s styles with volume at the temples work well.',
    }
    return descriptions[shape] || 'Your unique face shape allows for various men\'s hairstyle options.'
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getConfidenceText = (confidence: number) => {
    if (confidence >= 0.8) return 'High Confidence'
    if (confidence >= 0.6) return 'Moderate Confidence'
    return 'Low Confidence'
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Face Analysis Complete</h2>
        <p className="text-gray-600">Your facial features have been analyzed for optimal men's hairstyle recommendations</p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        {/* Analyzed Image */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-800">Facial Feature Detection</h3>
          <div className="bg-gray-50 rounded-xl p-4">
            <canvas
              ref={canvasRef}
              className="max-w-full h-auto rounded-lg border border-gray-200"
            />
          </div>
          <p className="text-sm text-gray-500 text-center">
            Blue dots indicate detected facial landmarks used for analysis
          </p>
        </div>

        {/* Analysis Results */}
        <div className="space-y-6">
          <div className="bg-primary-50 rounded-xl p-6 border border-primary-200">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">Analysis Results</h3>
            
            <div className="space-y-4">
              {/* Face Shape */}
              <div className="flex justify-between items-center">
                <span className="text-gray-700">Face Shape:</span>
                <span className="font-semibold text-primary-700 text-lg">{analysis.face_shape}</span>
              </div>

              {/* Confidence */}
              <div className="flex justify-between items-center">
                <span className="text-gray-700">Analysis Confidence:</span>
                <span className={`font-medium ${getConfidenceColor(analysis.confidence)}`}>
                  {getConfidenceText(analysis.confidence)} ({Math.round(analysis.confidence * 100)}%)
                </span>
              </div>

              {/* Landmarks Count */}
              <div className="flex justify-between items-center">
                <span className="text-gray-700">Facial Points Detected:</span>
                <span className="font-medium text-gray-800">{analysis.landmarks?.length || 0}</span>
              </div>
            </div>
          </div>

          {/* Face Shape Description */}
          <div className="bg-white border border-gray-200 rounded-xl p-6">
            <h4 className="font-semibold text-gray-800 mb-3">About Your {analysis.face_shape} Face Shape</h4>
            <p className="text-gray-600 text-sm leading-relaxed">
              {getFaceShapeDescription(analysis.face_shape)}
            </p>
          </div>

          {/* Next Steps */}
          <div className="bg-accent-50 border border-accent-200 rounded-xl p-6">
            <h4 className="font-semibold text-accent-800 mb-3">What's Next?</h4>
            <p className="text-accent-700 text-sm mb-4">
              Based on your face shape analysis, we'll show you men's hairstyles that will:
            </p>
            <ul className="text-accent-700 text-sm space-y-1">
              <li>• Complement your facial structure</li>
              <li>• Enhance your best features</li>
              <li>• Provide modern, masculine styling options</li>
              <li>• Suit your lifestyle and preferences</li>
            </ul>
          </div>

          {/* Continue Button */}
          <button
            onClick={onContinue}
            className="w-full bg-primary-600 text-white font-medium py-4 rounded-xl hover:bg-primary-700 transition-colors flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
            </svg>
            <span>See Recommended Men's Hairstyles</span>
          </button>
        </div>
      </div>
    </div>
  )
} 