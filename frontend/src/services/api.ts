import axios from 'axios'
import {
  FaceAnalysisResponse,
  HairstyleResponse,
  HairstyleRecommendationRequest,
  ImageGenerationRequest,
  ImageGenerationResponse,
  UserProfileRequest,
  UserProfileResponse,
  ProfileOptions,
  EthnicityOption,
  HairTextureOption,
} from '@/types/api'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const apiService = {
  // User Profile Management
  createProfile: async (profile: UserProfileRequest): Promise<UserProfileResponse> => {
    const response = await api.post('/create-profile/', profile)
    return response.data
  },

  getProfile: async (profileId: string): Promise<UserProfileResponse> => {
    const response = await api.get(`/profile/${profileId}`)
    return response.data
  },

  updateProfile: async (profileId: string, profile: UserProfileRequest): Promise<UserProfileResponse> => {
    const response = await api.put(`/profile/${profileId}`, profile)
    return response.data
  },

  getProfileOptions: async (): Promise<ProfileOptions> => {
    const response = await api.get('/profile-options/')
    return response.data
  },

  getEthnicities: async (): Promise<{ ethnicities: EthnicityOption[] }> => {
    const response = await api.get('/ethnicities/')
    return response.data
  },

  getHairTextures: async (): Promise<{ hair_textures: HairTextureOption[] }> => {
    const response = await api.get('/hair-textures/')
    return response.data
  },

  // Analyze face in uploaded images
  analyzeFace: async (files: File[], profileId?: string): Promise<FaceAnalysisResponse> => {
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })

    const url = profileId ? `/analyze-face/?profile_id=${profileId}` : '/analyze-face/'
    const response = await api.post(url, formData, {
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