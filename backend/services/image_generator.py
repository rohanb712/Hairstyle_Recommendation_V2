import os
import httpx
import base64
import json
from typing import List, Optional
from models.schemas import HairstyleResponse

class ImageGenerator:
    """Service for generating AI images with hairstyles"""
    
    def __init__(self):
        # Configuration for external AI service
        self.api_key = os.getenv('STABILITY_API_KEY') or os.getenv('OPENAI_API_KEY')
        self.api_provider = os.getenv('AI_PROVIDER', 'stability')  # 'stability' or 'openai'
        
        # API endpoints
        self.stability_url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
        self.openai_url = "https://api.openai.com/v1/images/generations"
        
    async def generate_image(self, user_images: List[str], hairstyle: HairstyleResponse) -> str:
        """
        Generate an AI image of the user with the selected hairstyle
        
        Args:
            user_images: List of base64 encoded user images or URLs
            hairstyle: Selected hairstyle object
            
        Returns:
            URL or base64 of the generated image
        """
        try:
            # Construct the prompt
            prompt = self._construct_prompt(hairstyle)
            
            # For V1, we'll use a simple text-to-image approach
            # In production, you'd want to use IP-Adapter or similar for better resemblance
            
            if self.api_provider == 'stability' and self.api_key:
                return await self._generate_with_stability(prompt)
            elif self.api_provider == 'openai' and self.api_key:
                return await self._generate_with_openai(prompt)
            else:
                # Fallback: return a placeholder image URL
                return await self._generate_placeholder_image(hairstyle)
                
        except Exception as e:
            print(f"Error generating image: {e}")
            return await self._generate_placeholder_image(hairstyle)
    
    def _construct_prompt(self, hairstyle: HairstyleResponse) -> str:
        """Construct the AI generation prompt"""
        base_prompt = "Portrait of a person"
        style_prompt = hairstyle.generation_prompt_modifier
        quality_prompt = "high quality, professional photography, studio lighting, detailed"
        
        full_prompt = f"{base_prompt} {style_prompt}, {quality_prompt}"
        return full_prompt
    
    async def _generate_with_stability(self, prompt: str) -> str:
        """Generate image using Stability AI API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "text_prompts": [
                    {
                        "text": prompt,
                        "weight": 1
                    }
                ],
                "cfg_scale": 7,
                "height": 1024,
                "width": 1024,
                "samples": 1,
                "steps": 30,
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.stability_url,
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Return base64 image data
                    image_data = result["artifacts"][0]["base64"]
                    return f"data:image/png;base64,{image_data}"
                else:
                    print(f"Stability AI API error: {response.status_code}")
                    return await self._generate_placeholder_image(None)
                    
        except Exception as e:
            print(f"Error with Stability AI: {e}")
            return await self._generate_placeholder_image(None)
    
    async def _generate_with_openai(self, prompt: str) -> str:
        """Generate image using OpenAI DALL-E API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            data = {
                "prompt": prompt,
                "n": 1,
                "size": "1024x1024",
                "response_format": "url"
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.openai_url,
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["data"][0]["url"]
                else:
                    print(f"OpenAI API error: {response.status_code}")
                    return await self._generate_placeholder_image(None)
                    
        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return await self._generate_placeholder_image(None)
    
    async def _generate_placeholder_image(self, hairstyle: Optional[HairstyleResponse]) -> str:
        """Generate a placeholder image for demo purposes"""
        # For V1, return a placeholder image URL
        # In production, you might want to generate a simple image or use a default
        
        placeholder_images = {
            "pixie_cut": "https://images.unsplash.com/photo-1594736797933-d0401ba2fe65?w=400",
            "bob_cut": "https://images.unsplash.com/photo-1580618672591-eb180b1a973f?w=400",
            "long_layers": "https://images.unsplash.com/photo-1616683693504-3ea7e9ad6fec?w=400",
            "side_swept_bangs": "https://images.unsplash.com/photo-1598300042247-d088f8ab3a91?w=400",
            "beach_waves": "https://images.unsplash.com/photo-1605497788044-5a32c7078486?w=400",
            "blunt_cut": "https://images.unsplash.com/photo-1580618672591-eb180b1a973f?w=400",
            "curly_shag": "https://images.unsplash.com/photo-1616683693504-3ea7e9ad6fec?w=400",
            "updo_bun": "https://images.unsplash.com/photo-1594736797933-d0401ba2fe65?w=400"
        }
        
        if hairstyle and hairstyle.id in placeholder_images:
            return placeholder_images[hairstyle.id]
        
        # Default placeholder
        return "https://images.unsplash.com/photo-1580618672591-eb180b1a973f?w=400" 