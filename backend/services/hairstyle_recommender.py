import json
import os
from typing import List, Optional, Dict, Any
from models.schemas import HairstyleResponse

class HairstyleRecommender:
    """Service for recommending hairstyles based on face shape"""
    
    def __init__(self):
        self.hairstyles_data = []
        self._load_hairstyles()
    
    def _load_hairstyles(self):
        """Load hairstyle data from JSON file"""
        try:
            # Get the path to the data file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(current_dir, '..', 'data', 'hairstyles.json')
            
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    self.hairstyles_data = json.load(f)
            else:
                # If file doesn't exist, create default data
                self._create_default_hairstyles()
                
        except Exception as e:
            print(f"Error loading hairstyles: {e}")
            self._create_default_hairstyles()
    
    def _create_default_hairstyles(self):
        """Create default hairstyle data for demo purposes"""
        self.hairstyles_data = [
            {
                "id": "pixie_cut",
                "name": "Pixie Cut",
                "description": "A short, cropped hairstyle that's chic and low-maintenance. Perfect for highlighting facial features.",
                "image_url": "https://example.com/pixie_cut.jpg",
                "suitable_face_shapes": ["Oval", "Heart", "Square"],
                "generation_prompt_modifier": "with a stylish short pixie cut hairstyle, cropped hair, modern and chic"
            },
            {
                "id": "bob_cut",
                "name": "Bob Cut",
                "description": "A classic bob that falls between the chin and shoulders. Versatile and timeless.",
                "image_url": "https://example.com/bob_cut.jpg",
                "suitable_face_shapes": ["Oval", "Round", "Square"],
                "generation_prompt_modifier": "with a classic bob haircut, shoulder-length hair, sleek and professional"
            },
            {
                "id": "long_layers",
                "name": "Long Layered Hair",
                "description": "Long hair with layers that add volume and movement. Great for creating dimension.",
                "image_url": "https://example.com/long_layers.jpg",
                "suitable_face_shapes": ["Long", "Round", "Square"],
                "generation_prompt_modifier": "with long layered hair, flowing layers, voluminous and bouncy"
            },
            {
                "id": "side_swept_bangs",
                "name": "Side-Swept Bangs",
                "description": "Soft, side-swept bangs that frame the face beautifully. Can be paired with various lengths.",
                "image_url": "https://example.com/side_swept_bangs.jpg",
                "suitable_face_shapes": ["Heart", "Long", "Square"],
                "generation_prompt_modifier": "with side-swept bangs, soft fringe, face-framing layers"
            },
            {
                "id": "beach_waves",
                "name": "Beach Waves",
                "description": "Relaxed, tousled waves that give a natural, effortless look. Perfect for a casual style.",
                "image_url": "https://example.com/beach_waves.jpg",
                "suitable_face_shapes": ["Oval", "Round", "Heart"],
                "generation_prompt_modifier": "with beachy waves, tousled hair, natural wavy texture, effortless style"
            },
            {
                "id": "blunt_cut",
                "name": "Blunt Cut",
                "description": "A straight-across cut that creates a bold, modern look. Works well with straight hair.",
                "image_url": "https://example.com/blunt_cut.jpg",
                "suitable_face_shapes": ["Oval", "Heart"],
                "generation_prompt_modifier": "with a blunt cut hairstyle, straight across cut, sharp and modern"
            },
            {
                "id": "curly_shag",
                "name": "Curly Shag",
                "description": "A textured, layered cut that enhances natural curls. Adds volume and movement.",
                "image_url": "https://example.com/curly_shag.jpg",
                "suitable_face_shapes": ["Round", "Square", "Long"],
                "generation_prompt_modifier": "with a curly shag hairstyle, textured layers, voluminous curls, bouncy texture"
            },
            {
                "id": "updo_bun",
                "name": "Elegant Updo",
                "description": "A sophisticated updo perfect for formal occasions. Showcases the neck and facial features.",
                "image_url": "https://example.com/updo_bun.jpg",
                "suitable_face_shapes": ["Oval", "Long", "Heart"],
                "generation_prompt_modifier": "with an elegant updo hairstyle, hair pulled back, sophisticated bun, formal style"
            }
        ]
    
    async def get_recommendations(self, face_shape: str) -> List[HairstyleResponse]:
        """
        Get hairstyle recommendations for a given face shape
        
        Args:
            face_shape: The classified face shape
            
        Returns:
            List of recommended hairstyles
        """
        try:
            recommendations = []
            
            for hairstyle in self.hairstyles_data:
                if face_shape in hairstyle.get('suitable_face_shapes', []):
                    recommendations.append(HairstyleResponse(
                        id=hairstyle['id'],
                        name=hairstyle['name'],
                        description=hairstyle['description'],
                        image_url=hairstyle['image_url'],
                        suitable_face_shapes=hairstyle['suitable_face_shapes'],
                        generation_prompt_modifier=hairstyle['generation_prompt_modifier']
                    ))
            
            # If no specific recommendations found, return some general ones
            if not recommendations and self.hairstyles_data:
                # Return first 3 hairstyles as fallback
                for hairstyle in self.hairstyles_data[:3]:
                    recommendations.append(HairstyleResponse(
                        id=hairstyle['id'],
                        name=hairstyle['name'],
                        description=hairstyle['description'],
                        image_url=hairstyle['image_url'],
                        suitable_face_shapes=hairstyle['suitable_face_shapes'],
                        generation_prompt_modifier=hairstyle['generation_prompt_modifier']
                    ))
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return []
    
    async def get_hairstyle_by_id(self, hairstyle_id: str) -> Optional[HairstyleResponse]:
        """
        Get a specific hairstyle by ID
        
        Args:
            hairstyle_id: The ID of the hairstyle
            
        Returns:
            HairstyleResponse object or None if not found
        """
        try:
            for hairstyle in self.hairstyles_data:
                if hairstyle['id'] == hairstyle_id:
                    return HairstyleResponse(
                        id=hairstyle['id'],
                        name=hairstyle['name'],
                        description=hairstyle['description'],
                        image_url=hairstyle['image_url'],
                        suitable_face_shapes=hairstyle['suitable_face_shapes'],
                        generation_prompt_modifier=hairstyle['generation_prompt_modifier']
                    )
            
            return None
            
        except Exception as e:
            print(f"Error getting hairstyle by ID: {e}")
            return None 