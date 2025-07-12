#!/usr/bin/env python3
"""
Test script to verify Gemini image generation integration works correctly
"""
import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add backend to path
sys.path.append('backend')

from services.llm_hairstyle_recommender import LLMHairstyleRecommender

async def test_image_generation():
    """Test the LLM hairstyle recommender with image generation"""
    
    # Check if API key is available
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("   Please set your Gemini API key:")
        print("   export GEMINI_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ Found Gemini API key: {api_key[:8]}...")
    
    try:
        # Initialize service
        recommender = LLMHairstyleRecommender(api_key)
        print("✅ LLMHairstyleRecommender initialized")
        
        # Test direct image generation
        print("\n🖼️  Testing direct image generation...")
        test_prompt = "Professional portrait of a person with modern fade haircut, high quality, studio lighting"
        
        image_url = await recommender.generate_hairstyle_image(test_prompt)
        
        if image_url:
            print(f"✅ Generated image successfully!")
            print(f"   Image URL: {image_url[:50]}...")
        else:
            print("❌ Failed to generate image")
            return False
        
        # Test full recommendation with image generation
        print("\n📋 Testing full recommendation with image generation...")
        recommendations = await recommender.get_recommendations(
            face_shape="oval",
            gender="male",
            age=25,
            race="caucasian",
            additional_context="Modern professional look",
            max_results=2  # Only 2 to save time/cost
        )
        
        if recommendations:
            print(f"✅ Generated {len(recommendations)} recommendations with images")
            for i, rec in enumerate(recommendations):
                print(f"   {i+1}. {rec.name}")
                if rec.image_url.startswith("data:image"):
                    print(f"      ✅ Generated image available")
                else:
                    print(f"      ⚠️  Using fallback image: {rec.image_url[:50]}...")
        else:
            print("❌ No recommendations generated")
            return False
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("🧪 GEMINI IMAGE GENERATION TEST")
    print("="*60)
    
    success = asyncio.run(test_image_generation())
    
    if success:
        print("\n✅ Integration test passed!")
        print("   Your Gemini image generation is working correctly.")
        print("   You can now start the backend to use AI-generated images.")
    else:
        print("\n❌ Integration test failed!")
        print("   Please check your setup and try again.") 