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
    HairstyleResponse,
    ImageGenerationRequest,
    ImageGenerationResponse
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="AI Hairstyle Recommender & Visualizer")

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
    return {"status": "healthy", "service": "AI Hairstyle Recommender & Visualizer"}

@app.post("/analyze-face/", response_model=FaceAnalysisResponse)
async def analyze_face(files: List[UploadFile] = File(...)):
    """
    Analyze faces in uploaded images and classify face shape.
    Returns detected facial landmarks and classified face shape.
    """
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files uploaded")
        
        # Process the first file for V1
        file = files[0]
        
        # Read file contents
        contents = await file.read()
        
        # Process image and detect landmarks
        landmarks, aligned_face = await image_processor.process_image(contents)
        
        if landmarks is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Classify face shape
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
    Get hairstyle recommendations based on face shape.
    Returns a list of suitable hairstyles.
    """
    try:
        recommendations = await hairstyle_recommender.get_recommendations(
            face_shape=request.face_shape
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"No hairstyles found for face shape: {request.face_shape}"
            )
        
        return recommendations
    
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
        
        # Generate image
        generated_image_url = await image_generator.generate_image(
            user_images=request.user_images,
            hairstyle=hairstyle
        )
        
        return ImageGenerationResponse(
            image_url=generated_image_url,
            prompt_used=hairstyle.generation_prompt_modifier
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 