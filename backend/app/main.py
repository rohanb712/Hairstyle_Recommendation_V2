from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

from services.image_processor import ImageProcessor
from services.face_classifier import FaceClassifier
from services.hairstyle_recommender import HairstyleRecommender
from services.image_generator import ImageGenerator
from models.schemas import (
    FaceAnalysisResponse,
    HairstyleRecommendationRequest,
    GenderBasedRecommendationRequest,
    HairstyleResponse,
    ImageGenerationRequest,
    ImageGenerationResponse,
    HairstyleStatsResponse
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
image_processor = ImageProcessor()
face_classifier = FaceClassifier()
hairstyle_recommender = HairstyleRecommender()
image_generator = ImageGenerator()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "service": "AI Hairstyle Recommender & Visualizer",
        "version": "2.0.0",
        "features": [
            "VGG16-based face shape classification",
            "MTCNN face detection",
            "Gender-specific recommendations",
            "Male and female hairstyles"
        ]
    }

@app.post("/analyze-face/", response_model=FaceAnalysisResponse)
async def analyze_face(files: List[UploadFile] = File(...)):
    """
    Analyze faces in uploaded images and classify face shape using VGG16 model.
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
        
        return FaceAnalysisResponse(
            landmarks=landmarks,
            face_shape=face_shape,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-hairstyles/", response_model=List[HairstyleResponse])
async def recommend_hairstyles(request: HairstyleRecommendationRequest):
    """
    Get hairstyle recommendations based on face shape and optional gender.
    
    Face shapes supported: oval, round, square, heart, long
    Gender options: "male", "female", or None for all styles
    """
    try:
        recommendations = await hairstyle_recommender.get_recommendations(
            face_shape=request.face_shape,
            gender=request.gender,
            max_results=6
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"No hairstyles found for face shape: {request.face_shape}" + 
                       (f" and gender: {request.gender}" if request.gender else "")
            )
        
        return recommendations
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        # Get hairstyle details
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 