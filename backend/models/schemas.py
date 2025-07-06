from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class EthnicityEnum(str, Enum):
    """Supported ethnicities"""
    AFRICAN = "african"
    ASIAN = "asian"
    CAUCASIAN = "caucasian"
    HISPANIC = "hispanic"
    MIDDLE_EASTERN = "middle_eastern"
    NATIVE_AMERICAN = "native_american"
    PACIFIC_ISLANDER = "pacific_islander"
    MIXED = "mixed"
    OTHER = "other"

class HairTextureEnum(str, Enum):
    """Hair texture types"""
    STRAIGHT = "straight"
    WAVY = "wavy"
    CURLY = "curly"
    COILY = "coily"

class GenderEnum(str, Enum):
    """Gender options"""
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"

class UserProfile(BaseModel):
    """User profile with demographic and hair information"""
    ethnicity: EthnicityEnum
    age: int = Field(ge=13, le=100, description="Age must be between 13 and 100")
    gender: GenderEnum
    hair_texture: HairTextureEnum
    face_shape: Optional[str] = None  # Will be populated after analysis
    
    class Config:
        use_enum_values = True

class FaceAnalysisRequest(BaseModel):
    """Request model for face analysis with user profile"""
    user_profile: Optional[UserProfile] = None

class FaceAnalysisResponse(BaseModel):
    """Response model for face analysis endpoint"""
    landmarks: List[List[float]]  # List of [x, y] coordinates
    face_shape: str
    confidence: float
    user_profile: Optional[UserProfile] = None

class HairstyleRecommendationRequest(BaseModel):
    """Request model for hairstyle recommendations with enhanced user profile"""
    face_shape: str
    user_profile: Optional[UserProfile] = None
    # Legacy support for simple gender-only requests
    gender: Optional[str] = None  # "male", "female", or None for all

class GenderBasedRecommendationRequest(BaseModel):
    """Request model for gender-only based recommendations"""
    gender: str  # "male" or "female"
    
class UserProfileRequest(BaseModel):
    """Request model for creating/updating user profile"""
    ethnicity: EthnicityEnum
    age: int = Field(ge=13, le=100, description="Age must be between 13 and 100")
    gender: GenderEnum
    hair_texture: HairTextureEnum
    
class UserProfileResponse(BaseModel):
    """Response model for user profile operations"""
    profile_id: str
    user_profile: UserProfile
    created_at: str
    updated_at: str

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