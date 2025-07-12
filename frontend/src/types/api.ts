// User Profile Types
export type EthnicityType = 'african' | 'asian' | 'caucasian' | 'hispanic' | 'middle_eastern' | 'native_american' | 'pacific_islander' | 'south_asian' | 'mixed' | 'other'
export type HairTextureType = 'straight' | 'wavy' | 'curly' | 'coily'
export type GenderType = 'male' | 'female' | 'non_binary'

export interface UserProfile {
  ethnicity: EthnicityType
  age: number
  gender: GenderType
  hair_texture: HairTextureType
  face_shape?: string
}

export interface UserProfileRequest {
  ethnicity: EthnicityType
  age: number
  gender: GenderType
  hair_texture: HairTextureType
}

export interface UserProfileResponse {
  profile_id: string
  user_profile: UserProfile
  created_at: string
  updated_at: string
}

export interface ProfileOptions {
  ethnicities: EthnicityType[]
  hair_textures: HairTextureType[]
  genders: GenderType[]
  age_range: {
    min: number
    max: number
  }
}

export interface EthnicityOption {
  value: EthnicityType
  label: string
}

export interface HairTextureOption {
  value: HairTextureType
  label: string
}

// Updated existing types
export interface FaceAnalysisResponse {
  landmarks: number[][]
  face_shape: string
  confidence: number
  user_profile?: UserProfile
}

export interface HairstyleResponse {
  id: string
  name: string
  description: string
  image_url: string
  suitable_face_shapes: string[]
  generation_prompt_modifier: string
}

export interface HairstyleRecommendationRequest {
  face_shape: string
  user_profile?: UserProfile
  gender?: string // Legacy support
}

export interface ImageGenerationRequest {
  user_images: string[]
  hairstyle_id: string
  gender?: string
}

export interface ImageGenerationResponse {
  image_url: string
  prompt_used: string
} 