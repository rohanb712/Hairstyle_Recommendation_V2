from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class FaceAnalysisResponse(BaseModel):
    """Response model for face analysis endpoint"""
    landmarks: List[List[float]]  # List of [x, y] coordinates
    face_shape: str
    confidence: float

class HairstyleRecommendationRequest(BaseModel):
    """Request model for hairstyle recommendations"""
    face_shape: str
    gender: Optional[str] = None  # "male", "female", or None for all

class GenderBasedRecommendationRequest(BaseModel):
    """Request model for gender-only based recommendations"""
    gender: str  # "male" or "female"

class HairstyleResponse(BaseModel):
    """Response model for individual hairstyle"""
    id: str
    name: str
    description: str
    image_url: str
    suitable_face_shapes: List[str]
    generation_prompt_modifier: str

class ImageGenerationRequest(BaseModel):
    """Request model for image generation"""
    user_images: List[str]  # Base64 encoded images or URLs
    hairstyle_id: str
    gender: Optional[str] = None  # For better prompt generation

class ImageGenerationResponse(BaseModel):
    """Response model for image generation"""
    image_url: str
    prompt_used: str

class HairstyleStatsResponse(BaseModel):
    """Response model for hairstyle database statistics"""
    total_hairstyles: int
    by_gender: Dict[str, int]
    by_face_shape: Dict[str, int]
    available_face_shapes: List[str]
    available_genders: List[str] 