import os
import json
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from models.schemas import HairstyleResponse


class HairstyleRecommendation(BaseModel):
    """Structured model for a single hairstyle recommendation"""
    id: str = Field(description="Unique identifier for the hairstyle (snake_case)")
    name: str = Field(description="Display name of the hairstyle")
    description: str = Field(description="Detailed description explaining why this style works")
    image_url: str = Field(description="Placeholder image URL from Unsplash")
    suitable_face_shapes: List[str] = Field(description="List of face shapes this style suits")
    generation_prompt_modifier: str = Field(description="Prompt modifier for AI image generation")


class HairstyleRecommendationsOutput(BaseModel):
    """Output model for multiple hairstyle recommendations"""
    recommendations: List[HairstyleRecommendation] = Field(
        description="List of 6 hairstyle recommendations"
    )


class HairstyleOutputParser(BaseOutputParser[HairstyleRecommendationsOutput]):
    """Custom parser for hairstyle recommendations JSON output"""
    
    def parse(self, text: str) -> HairstyleRecommendationsOutput:
        try:
            # Extract JSON from the response if it's wrapped in markdown code blocks
            if "```json" in text:
                start_idx = text.find("```json") + 7
                end_idx = text.find("```", start_idx)
                json_text = text[start_idx:end_idx].strip()
            elif "```" in text:
                start_idx = text.find("```") + 3
                end_idx = text.find("```", start_idx)
                json_text = text[start_idx:end_idx].strip()
            else:
                json_text = text.strip()
            
            # Parse the JSON
            data = json.loads(json_text)
            return HairstyleRecommendationsOutput(**data)
        except Exception as e:
            # Fallback to a basic response if parsing fails
            fallback_recommendations = [
                {
                    "id": "classic_recommended",
                    "name": "Classic Style",
                    "description": "A versatile style that complements your features.",
                    "image_url": "https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400",
                    "suitable_face_shapes": ["oval", "round", "square"],
                    "generation_prompt_modifier": "with a classic hairstyle, well-groomed appearance"
                }
            ]
            return HairstyleRecommendationsOutput(recommendations=fallback_recommendations)


class LLMHairstyleRecommender:
    """AI-powered hairstyle recommendation service using Google Gemini"""
    
    def __init__(self, api_key: str):
        """Initialize the LLM service with Google API key"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        self.output_parser = HairstyleOutputParser()
        self._setup_prompt_template()
    
    def _setup_prompt_template(self):
        """Set up the prompt template for hairstyle recommendations"""
        
        template = """You are a professional hairstylist AI assistant specializing in personalized hairstyle recommendations. 
Analyze the provided user characteristics and recommend 6 diverse hairstyles that would look excellent on them.

USER CHARACTERISTICS:
- Face Shape: {face_shape}
- Gender: {gender}
- Age: {age}
- Race/Ethnicity: {race}
- Additional Context: {additional_context}

REQUIREMENTS:
1. Provide exactly 6 diverse hairstyle recommendations
2. Each recommendation must include practical reasoning for why it suits the user
3. Consider cultural appropriateness and authenticity
4. Include both trendy and classic options
5. Account for hair texture and maintenance preferences when possible

OUTPUT FORMAT:
Respond with a JSON object matching this exact structure:

```json
{{
  "recommendations": [
    {{
      "id": "unique_hairstyle_id_snake_case",
      "name": "Hairstyle Display Name",
      "description": "One concise sentence explaining why this style works well for the user's face shape and features.",
      "image_url": "https://images.unsplash.com/photo-XXXXXXXXX?w=400",
      "suitable_face_shapes": ["face_shape1", "face_shape2"],
      "generation_prompt_modifier": "with specific hairstyle description for AI image generation, include texture and styling details"
    }}
  ]
}}
```

IMPORTANT GUIDELINES:
- Use realistic Unsplash photo URLs (you can use existing hairstyle photo IDs or similar)
- Make IDs unique and descriptive (e.g., "modern_fade_textured", "long_layered_waves")
- Face shapes: oval, round, square, heart, long
- Generation prompts should be detailed enough for AI image generation
- Descriptions should be personalized to the specific user characteristics
- Consider hair growth patterns and face proportions
- Include both low-maintenance and styled options

Generate the JSON response now:"""

        self.prompt = PromptTemplate(
            input_variables=["face_shape", "gender", "age", "race", "additional_context"],
            template=template
        )
    
    async def get_recommendations(self, 
                                face_shape: str, 
                                gender: Optional[str] = None,
                                age: Optional[int] = None,
                                race: Optional[str] = None,
                                additional_context: Optional[str] = None,
                                max_results: int = 6) -> List[HairstyleResponse]:
        """
        Get AI-powered hairstyle recommendations based on user characteristics
        
        Args:
            face_shape: The classified face shape (oval, round, square, heart, long)
            gender: User's gender ("male", "female", or other)
            age: User's age
            race: User's race/ethnicity
            additional_context: Additional preferences or context
            max_results: Maximum number of recommendations (default 6)
            
        Returns:
            List of recommended hairstyles
        """
        try:
            # Prepare input data
            prompt_input = {
                "face_shape": face_shape.lower(),
                "gender": gender or "any",
                "age": str(age) if age else "not specified",
                "race": race or "not specified", 
                "additional_context": additional_context or "General styling preferences"
            }
            
            # Create the prompt
            formatted_prompt = self.prompt.format(**prompt_input)
            
            # Get LLM response
            response = await self.llm.ainvoke(formatted_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse the response
            parsed_output = self.output_parser.parse(response_text)
            
            # Convert to HairstyleResponse objects
            recommendations = []
            for rec in parsed_output.recommendations[:max_results]:
                recommendations.append(HairstyleResponse(
                    id=rec.id,
                    name=rec.name,
                    description=rec.description,
                    image_url=rec.image_url,
                    suitable_face_shapes=rec.suitable_face_shapes,
                    generation_prompt_modifier=rec.generation_prompt_modifier
                ))
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting LLM recommendations: {e}")
            # Return fallback recommendations
            return await self._get_fallback_recommendations(face_shape, gender, max_results)
    
    async def _get_fallback_recommendations(self, 
                                         face_shape: str, 
                                         gender: Optional[str] = None,
                                         max_results: int = 6) -> List[HairstyleResponse]:
        """Provide fallback recommendations if LLM fails"""
        
        # Basic fallback based on face shape and gender
        fallbacks = {
            "male": [
                {
                    "id": "classic_fade_fallback",
                    "name": "Classic Fade",
                    "description": "A timeless men's cut with short sides that gradually fade into longer hair on top. Professional and versatile for any occasion.",
                    "image_url": "https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400",
                    "suitable_face_shapes": ["oval", "round", "square"],
                    "generation_prompt_modifier": "with a classic fade haircut, short sides, well-groomed men's hairstyle"
                },
                {
                    "id": "textured_crop_fallback", 
                    "name": "Textured Crop",
                    "description": "Modern short cut with textured, messy styling on top. Casual yet put-together appearance.",
                    "image_url": "https://images.unsplash.com/photo-1582095133179-bfd08e2fc6b3?w=400",
                    "suitable_face_shapes": ["oval", "round", "square"],
                    "generation_prompt_modifier": "with a textured crop hairstyle, messy textured top, modern casual men's cut"
                }
            ],
            "female": [
                {
                    "id": "bob_cut_fallback",
                    "name": "Classic Bob",
                    "description": "A timeless bob that falls between the chin and shoulders. Versatile and elegant for professional and casual settings.",
                    "image_url": "https://images.unsplash.com/photo-1580618672591-eb180b1a973f?w=400",
                    "suitable_face_shapes": ["oval", "round", "square", "long"],
                    "generation_prompt_modifier": "with a classic bob haircut, shoulder-length hair, sleek and professional"
                },
                {
                    "id": "long_layers_fallback",
                    "name": "Long Layered Hair", 
                    "description": "Long hair with layers that add volume and movement. Great for creating dimension and softening angular features.",
                    "image_url": "https://images.unsplash.com/photo-1616683693504-3ea7e9ad6fec?w=400",
                    "suitable_face_shapes": ["long", "round", "square", "heart"],
                    "generation_prompt_modifier": "with long layered hair, flowing layers, voluminous and bouncy"
                }
            ]
        }
        
        # Select appropriate fallbacks
        gender_key = gender.lower() if gender and gender.lower() in fallbacks else "male"
        selected_fallbacks = fallbacks[gender_key][:max_results]
        
        recommendations = []
        for fallback in selected_fallbacks:
            recommendations.append(HairstyleResponse(**fallback))
        
        return recommendations
    
    def get_available_face_shapes(self) -> List[str]:
        """Get list of supported face shapes"""
        return ["oval", "round", "square", "heart", "long"]
    
    def get_available_genders(self) -> List[str]:
        """Get list of supported genders"""
        return ["male", "female", "other"] 