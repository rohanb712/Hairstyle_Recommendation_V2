from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

from services.image_processor import ImageProcessor
from services.face_classifier import FaceClassifier
from services.hairstyle_recommender import HairstyleRecommender
from services.llm_hairstyle_recommender import LLMHairstyleRecommender
from services.image_generator import ImageGenerator
from models.schemas import (
    FaceAnalysisRequest,
    FaceAnalysisResponse,
    HairstyleRecommendationRequest,
    GenderBasedRecommendationRequest,
    HairstyleResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    HairstyleStatsResponse,
    UserProfile,
    UserProfileRequest,
    UserProfileResponse,
    EthnicityEnum,
    HairTextureEnum,
    GenderEnum
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Hairstyle Recommender & Visualizer", 
    description="Advanced face shape classification and hairstyle recommendation system using VGG16 transfer learning",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
print("🚀 Initializing AI Hairstyle Recommender services...")
image_processor = ImageProcessor()
print("✅ Image processor initialized")

print("🧠 Initializing face shape classifier...")
face_classifier = FaceClassifier()
print("✅ Face classifier initialized")

# Initialize LLM-based hairstyle recommender
gemini_api_key = os.getenv('GEMINI_API_KEY')
if gemini_api_key:
    try:
        llm_hairstyle_recommender = LLMHairstyleRecommender(gemini_api_key)
        print("✅ LLM hairstyle recommender initialized with Google Gemini")
        use_llm_recommendations = True
    except Exception as e:
        print(f"⚠️  Failed to initialize LLM service: {e}")
        print("   Falling back to traditional recommender...")
        hairstyle_recommender = HairstyleRecommender()
        print("✅ Traditional hairstyle recommender initialized")
        use_llm_recommendations = False
else:
    print("⚠️  GEMINI_API_KEY not found in environment variables")
    print("   Using traditional recommendation system...")
    hairstyle_recommender = HairstyleRecommender()
    print("✅ Traditional hairstyle recommender initialized")
    use_llm_recommendations = False

image_generator = ImageGenerator()
print("✅ Image generator initialized")

print("🎉 All services ready!")

# In-memory storage for user profiles (in production, use a database)
user_profiles: Dict[str, Dict] = {}

@app.get("/")
async def root():
    """Health check endpoint"""
    features = [
        "VGG16-based face shape classification",
        "MTCNN face detection",
        "Gender-specific recommendations",
        "Male and female hairstyles"
    ]
    
    if use_llm_recommendations:
        features.extend([
            "Google Gemini AI-powered recommendations",
            "Personalized styling based on age, race, and features",
            "Cultural-aware hairstyle suggestions",
            "Dynamic hairstyle generation"
        ])
    else:
        features.append("Traditional rule-based recommendations")
    
    return {
        "status": "healthy", 
        "service": "AI Hairstyle Recommender & Visualizer",
        "version": "2.1.0",
        "llm_enabled": use_llm_recommendations,
        "features": features
    }

@app.post("/create-profile/", response_model=UserProfileResponse)
async def create_user_profile(profile_request: UserProfileRequest):
    """
    Create a new user profile with demographic and hair information.
    Returns a profile ID for future reference.
    """
    try:
        # Generate unique profile ID
        profile_id = str(uuid.uuid4())
        
        # Create user profile
        user_profile = UserProfile(
            ethnicity=profile_request.ethnicity,
            age=profile_request.age,
            gender=profile_request.gender,
            hair_texture=profile_request.hair_texture
        )
        
        # Store profile with metadata
        current_time = datetime.now().isoformat()
        user_profiles[profile_id] = {
            "profile": user_profile.dict(),
            "created_at": current_time,
            "updated_at": current_time
        }
        
        return UserProfileResponse(
            profile_id=profile_id,
            user_profile=user_profile,
            created_at=current_time,
            updated_at=current_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/profile/{profile_id}", response_model=UserProfileResponse)
async def get_user_profile(profile_id: str):
    """
    Retrieve a user profile by ID.
    """
    try:
        if profile_id not in user_profiles:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile_data = user_profiles[profile_id]
        
        return UserProfileResponse(
            profile_id=profile_id,
            user_profile=UserProfile(**profile_data["profile"]),
            created_at=profile_data["created_at"],
            updated_at=profile_data["updated_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/profile/{profile_id}", response_model=UserProfileResponse)
async def update_user_profile(profile_id: str, profile_request: UserProfileRequest):
    """
    Update an existing user profile.
    """
    try:
        if profile_id not in user_profiles:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Update user profile
        user_profile = UserProfile(
            ethnicity=profile_request.ethnicity,
            age=profile_request.age,
            gender=profile_request.gender,
            hair_texture=profile_request.hair_texture,
            face_shape=user_profiles[profile_id]["profile"].get("face_shape")  # Keep existing face shape
        )
        
        # Update stored profile
        current_time = datetime.now().isoformat()
        user_profiles[profile_id]["profile"] = user_profile.dict()
        user_profiles[profile_id]["updated_at"] = current_time
        
        return UserProfileResponse(
            profile_id=profile_id,
            user_profile=user_profile,
            created_at=user_profiles[profile_id]["created_at"],
            updated_at=current_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-face/", response_model=FaceAnalysisResponse)
async def analyze_face(files: List[UploadFile] = File(...), profile_id: Optional[str] = None):
    """
    Analyze faces in uploaded images and classify face shape using VGG16 model.
    Optionally associate with a user profile.
    Returns detected facial landmarks and classified face shape with confidence.
    
    Supports face shapes: oval, round, square, heart, long
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Process the first file
        file = files[0]
        
        # Read file contents
        contents = await file.read()
        
        # Process image and detect landmarks
        landmarks, aligned_face = await image_processor.process_image(contents)
        
        if landmarks is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Classify face shape using VGG model
        face_shape, confidence = await face_classifier.classify_face(aligned_face)
        
        # Get user profile if provided
        user_profile = None
        if profile_id and profile_id in user_profiles:
            profile_data = user_profiles[profile_id]
            user_profile = UserProfile(**profile_data["profile"])
            
            # Update profile with detected face shape
            user_profile.face_shape = face_shape
            user_profiles[profile_id]["profile"]["face_shape"] = face_shape
            user_profiles[profile_id]["updated_at"] = datetime.now().isoformat()
        
        return FaceAnalysisResponse(
            landmarks=landmarks,
            face_shape=face_shape,
            confidence=confidence,
            user_profile=user_profile
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-hairstyles/", response_model=List[HairstyleResponse])
async def recommend_hairstyles(request: HairstyleRecommendationRequest):
    """
    Get AI-powered hairstyle recommendations based on face shape and user profile.
    
    Uses Google Gemini LLM for intelligent recommendations when available,
    otherwise falls back to traditional rule-based recommendations.
    
    Face shapes supported: oval, round, square, heart, long
    Gender options: "male", "female", or None for all styles
    """
    try:
        print(f"🔍 Recommendation request received:")
        print(f"   Face shape: {request.face_shape}")
        print(f"   User profile: {request.user_profile}")
        print(f"   Gender: {request.gender}")
        print(f"   Using LLM: {use_llm_recommendations}")
        
        # Use LLM-based recommendations if available
        if use_llm_recommendations:
            print("   Using LLM recommendations...")
            # Extract user characteristics from profile
            user_profile = request.user_profile
            
            print(f"   User profile type: {type(user_profile)}")
            print(f"   User profile dict: {user_profile.dict() if user_profile else None}")
            
            # Handle enum values that might already be strings
            if user_profile and user_profile.gender:
                print(f"   Gender type: {type(user_profile.gender)}")
                print(f"   Gender value: {user_profile.gender}")
                gender = user_profile.gender.value if hasattr(user_profile.gender, 'value') else user_profile.gender
            else:
                gender = request.gender
                
            age = user_profile.age if user_profile else None
            
            if user_profile and user_profile.ethnicity:
                print(f"   Ethnicity type: {type(user_profile.ethnicity)}")
                print(f"   Ethnicity value: {user_profile.ethnicity}")
                race = user_profile.ethnicity.value if hasattr(user_profile.ethnicity, 'value') else user_profile.ethnicity
            else:
                race = None
            
            print(f"   Extracted - Gender: {gender}, Age: {age}, Race: {race}")
            
            # Create additional context from hair texture and other profile data
            additional_context = ""
            if user_profile:
                if user_profile.hair_texture:
                    print(f"   Hair texture type: {type(user_profile.hair_texture)}")
                    print(f"   Hair texture value: {user_profile.hair_texture}")
                    hair_texture = user_profile.hair_texture.value if hasattr(user_profile.hair_texture, 'value') else user_profile.hair_texture
                    additional_context += f"Hair texture: {hair_texture}. "
                additional_context += "User prefers modern, stylish looks suitable for their lifestyle."
            
            print(f"   Additional context: {additional_context}")
            
            print("   Calling LLM recommender...")
            recommendations = await llm_hairstyle_recommender.get_recommendations(
                face_shape=request.face_shape,
                gender=gender,
                age=age,
                race=race,
                additional_context=additional_context if additional_context else None,
                max_results=6
            )
        else:
            print("   Using traditional recommendations...")
            # Fallback to traditional recommendations
            if request.user_profile and request.user_profile.gender:
                gender = request.user_profile.gender.value if hasattr(request.user_profile.gender, 'value') else request.user_profile.gender
            else:
                gender = request.gender
                
            recommendations = await hairstyle_recommender.get_recommendations(
                face_shape=request.face_shape,
                gender=gender,
                max_results=6
            )
        
        print(f"   Recommendations found: {len(recommendations) if recommendations else 0}")
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"No hairstyles found for face shape: {request.face_shape}" + 
                       (f" and gender: {gender}" if gender else "")
            )
        
        return recommendations
    
    except Exception as e:
        print(f"❌ Error in recommend_hairstyles: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

@app.post("/recommend-by-gender/", response_model=List[HairstyleResponse])
async def recommend_by_gender(request: GenderBasedRecommendationRequest):
    """
    Get hairstyle recommendations filtered by gender only.
    Useful for browsing all available styles for a specific gender.
    """
    try:
        recommendations = await hairstyle_recommender.get_recommendations_by_gender(
            gender=request.gender,
            max_results=10
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"No hairstyles found for gender: {request.gender}"
            )
        
        return recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hairstyle-stats/", response_model=HairstyleStatsResponse)
async def get_hairstyle_stats():
    """
    Get statistics about the hairstyle database including counts by gender and face shape.
    """
    try:
        stats = hairstyle_recommender.get_statistics()
        
        return HairstyleStatsResponse(
            total_hairstyles=stats['total_hairstyles'],
            by_gender=stats['by_gender'],
            by_face_shape=stats['by_face_shape'],
            available_face_shapes=stats['available_face_shapes'],
            available_genders=stats['available_genders']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-hairstyle-image/", response_model=ImageGenerationResponse)
async def generate_hairstyle_image(request: ImageGenerationRequest):
    """
    Generate an AI image of the user with the selected hairstyle.
    Returns URL or base64 of the generated image.
    """
    try:
        # Get hairstyle details - first try LLM store, then traditional store
        hairstyle = None
        
        # For LLM-generated hairstyles, we need to handle them differently
        # since they might not exist in the traditional hairstyle database
        if use_llm_recommendations:
            # The hairstyle_id from LLM recommendations should be passed directly
            # We'll create a mock hairstyle object for image generation
            # In a real implementation, you might want to store LLM results temporarily
            
            # For now, we'll try to get from traditional store first
            if not use_llm_recommendations:
                hairstyle = await hairstyle_recommender.get_hairstyle_by_id(request.hairstyle_id)
            
            # If not found and we're using LLM, create a basic hairstyle for generation
            if not hairstyle:
                # Extract info from request or use fallback
                from models.schemas import HairstyleResponse
                hairstyle = HairstyleResponse(
                    id=request.hairstyle_id,
                    name="Custom AI Style",
                    description="AI-recommended hairstyle based on your features",
                    image_url="https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400",
                    suitable_face_shapes=["oval", "round", "square", "heart", "long"],
                    generation_prompt_modifier=f"with a stylish {request.gender or 'modern'} hairstyle that complements facial features"
                )
        else:
            # Traditional lookup
            hairstyle = await hairstyle_recommender.get_hairstyle_by_id(request.hairstyle_id)
        
        if not hairstyle:
            raise HTTPException(
                status_code=404, 
                detail=f"Hairstyle not found: {request.hairstyle_id}"
            )
        
        # Generate image with enhanced prompt including gender
        generated_image_url = await image_generator.generate_image(
            user_images=request.user_images,
            hairstyle=hairstyle,
            gender=request.gender
        )
        
        return ImageGenerationResponse(
            image_url=generated_image_url,
            prompt_used=hairstyle.generation_prompt_modifier
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/face-shapes/")
async def get_available_face_shapes():
    """Get list of all supported face shapes"""
    return {
        "face_shapes": hairstyle_recommender.get_available_face_shapes(),
        "descriptions": {
            "oval": "Balanced proportions, face length greater than width",
            "round": "Face width approximately equals face length, soft curves",
            "square": "Strong angular jawline, face width approximately equals length",
            "heart": "Wide forehead, narrow chin, prominent cheekbones", 
            "long": "Face length significantly greater than width, narrow features"
        }
    }

@app.get("/genders/")
async def get_available_genders():
    """Get list of all supported genders for hairstyle filtering"""
    return {
        "genders": hairstyle_recommender.get_available_genders()
    }

@app.get("/profile-options/")
async def get_profile_options():
    """Get all available options for user profile creation"""
    return {
        "ethnicities": [e.value for e in EthnicityEnum],
        "hair_textures": [t.value for t in HairTextureEnum],
        "genders": [g.value for g in GenderEnum],
        "age_range": {"min": 13, "max": 100}
    }

@app.get("/ethnicities/")
async def get_available_ethnicities():
    """Get list of all supported ethnicities"""
    return {
        "ethnicities": [
            {"value": e.value, "label": e.value.replace("_", " ").title()}
            for e in EthnicityEnum
        ]
    }

@app.get("/hair-textures/")
async def get_available_hair_textures():
    """Get list of all supported hair textures"""
    return {
        "hair_textures": [
            {"value": t.value, "label": t.value.title()}
            for t in HairTextureEnum
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 