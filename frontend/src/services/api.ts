import axios from 'axios'
import {
  FaceAnalysisResponse,
  HairstyleResponse,
  HairstyleRecommendationRequest,
  ImageGenerationRequest,
  ImageGenerationResponse,
} from '@/types/api'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const apiService = {
  // Analyze face in uploaded images
  analyzeFace: async (files: File[]): Promise<FaceAnalysisResponse> => {
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })

    const response = await api.post('/analyze-face/', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    return response.data
  },

  // Get hairstyle recommendations based on face shape
  getHairstyleRecommendations: async (
    request: HairstyleRecommendationRequest
  ): Promise<HairstyleResponse[]> => {
    const response = await api.post('/recommend-hairstyles/', request)
    return response.data
  },

  // Generate AI image with selected hairstyle
  generateHairstyleImage: async (
    request: ImageGenerationRequest
  ): Promise<ImageGenerationResponse> => {
    const response = await api.post('/generate-hairstyle-image/', request)
    return response.data
  },

  // Health check
  healthCheck: async (): Promise<{ status: string; service: string }> => {
    const response = await api.get('/')
    return response.data
  },
}

export default apiService 