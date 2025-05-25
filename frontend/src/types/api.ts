export interface FaceAnalysisResponse {
  landmarks: number[][]
  face_shape: string
  confidence: number
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
}

export interface ImageGenerationRequest {
  user_images: string[]
  hairstyle_id: string
}

export interface ImageGenerationResponse {
  image_url: string
  prompt_used: string
} 