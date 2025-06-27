import json
import os
from typing import List, Optional, Dict, Any
from models.schemas import HairstyleResponse

class HairstyleRecommender:
    """Service for recommending hairstyles based on face shape and gender"""
    
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
                "image_url": "https://images.unsplash.com/photo-1594736797933-d0401ba2fe65?w=400",
                "suitable_face_shapes": ["oval", "heart", "square"],
                "gender": "female",
                "generation_prompt_modifier": "with a stylish short pixie cut hairstyle, cropped hair, modern and chic"
            },
            {
                "id": "classic_fade",
                "name": "Classic Fade",
                "description": "A timeless men's cut with short sides that gradually fade into longer hair on top. Professional and versatile.",
                "image_url": "https://images.unsplash.com/photo-1621605815971-fbc98d665033?w=400",
                "suitable_face_shapes": ["oval", "round", "square"],
                "gender": "male",
                "generation_prompt_modifier": "with a classic fade haircut, short sides, well-groomed men's hairstyle, professional appearance"
            },
            {
                "id": "bob_cut",
                "name": "Classic Bob",
                "description": "A timeless bob that falls between the chin and shoulders. Versatile and elegant.",
                "image_url": "https://images.unsplash.com/photo-1580618672591-eb180b1a973f?w=400",
                "suitable_face_shapes": ["oval", "round", "square", "long"],
                "gender": "female",
                "generation_prompt_modifier": "with a classic bob haircut, shoulder-length hair, sleek and professional"
            }
        ]
    
    async def get_recommendations(self, face_shape: str, gender: Optional[str] = None, 
                                max_results: int = 6) -> List[HairstyleResponse]:
        """
        Get hairstyle recommendations for a given face shape and optional gender
        
        Args:
            face_shape: The classified face shape (oval, round, square, heart, long)
            gender: Optional gender filter ("male" or "female")
            max_results: Maximum number of recommendations to return
            
        Returns:
            List of recommended hairstyles
        """
        try:
            recommendations = []
            
            # Normalize face shape input (handle both cases)
            face_shape_normalized = face_shape.lower()
            
            for hairstyle in self.hairstyles_data:
                # Check if face shape matches
                suitable_shapes = [shape.lower() for shape in hairstyle.get('suitable_face_shapes', [])]
                if face_shape_normalized not in suitable_shapes:
                    continue
                
                # Check gender filter if provided
                if gender:
                    hairstyle_gender = hairstyle.get('gender', 'unisex').lower()
                    if hairstyle_gender != 'unisex' and hairstyle_gender != gender.lower():
                        continue
                
                recommendations.append(HairstyleResponse(
                    id=hairstyle['id'],
                    name=hairstyle['name'],
                    description=hairstyle['description'],
                    image_url=hairstyle['image_url'],
                    suitable_face_shapes=hairstyle['suitable_face_shapes'],
                    generation_prompt_modifier=hairstyle['generation_prompt_modifier']
                ))
            
            # Limit results
            recommendations = recommendations[:max_results]
            
            # If no specific recommendations found, return some general ones
            if not recommendations and self.hairstyles_data:
                fallback_styles = []
                
                # Try to find styles that work for the face shape regardless of gender
                for hairstyle in self.hairstyles_data:
                    suitable_shapes = [shape.lower() for shape in hairstyle.get('suitable_face_shapes', [])]
                    if face_shape_normalized in suitable_shapes:
                        fallback_styles.append(hairstyle)
                
                # If still no matches, use styles that work for oval faces (most versatile)
                if not fallback_styles:
                    for hairstyle in self.hairstyles_data:
                        suitable_shapes = [shape.lower() for shape in hairstyle.get('suitable_face_shapes', [])]
                        if 'oval' in suitable_shapes:
                            fallback_styles.append(hairstyle)
                
                # Convert to response objects
                for hairstyle in fallback_styles[:max_results]:
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
    
    async def get_recommendations_by_gender(self, gender: str, max_results: int = 10) -> List[HairstyleResponse]:
        """
        Get hairstyle recommendations filtered by gender only
        
        Args:
            gender: Gender filter ("male" or "female")
            max_results: Maximum number of recommendations to return
            
        Returns:
            List of hairstyles for the specified gender
        """
        try:
            recommendations = []
            
            for hairstyle in self.hairstyles_data:
                hairstyle_gender = hairstyle.get('gender', 'unisex').lower()
                
                if hairstyle_gender == 'unisex' or hairstyle_gender == gender.lower():
                    recommendations.append(HairstyleResponse(
                        id=hairstyle['id'],
                        name=hairstyle['name'],
                        description=hairstyle['description'],
                        image_url=hairstyle['image_url'],
                        suitable_face_shapes=hairstyle['suitable_face_shapes'],
                        generation_prompt_modifier=hairstyle['generation_prompt_modifier']
                    ))
            
            return recommendations[:max_results]
            
        except Exception as e:
            print(f"Error getting gender-based recommendations: {e}")
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
    
    def get_available_face_shapes(self) -> List[str]:
        """Get list of all available face shapes from the data"""
        face_shapes = set()
        for hairstyle in self.hairstyles_data:
            for shape in hairstyle.get('suitable_face_shapes', []):
                face_shapes.add(shape.lower())
        return sorted(list(face_shapes))
    
    def get_available_genders(self) -> List[str]:
        """Get list of all available genders from the data"""
        genders = set()
        for hairstyle in self.hairstyles_data:
            gender = hairstyle.get('gender', 'unisex')
            genders.add(gender.lower())
        return sorted(list(genders))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the hairstyle database"""
        stats = {
            'total_hairstyles': len(self.hairstyles_data),
            'by_gender': {},
            'by_face_shape': {},
            'available_face_shapes': self.get_available_face_shapes(),
            'available_genders': self.get_available_genders()
        }
        
        # Count by gender
        for hairstyle in self.hairstyles_data:
            gender = hairstyle.get('gender', 'unisex')
            stats['by_gender'][gender] = stats['by_gender'].get(gender, 0) + 1
        
        # Count by face shape
        for hairstyle in self.hairstyles_data:
            for shape in hairstyle.get('suitable_face_shapes', []):
                shape_lower = shape.lower()
                stats['by_face_shape'][shape_lower] = stats['by_face_shape'].get(shape_lower, 0) + 1
        
        return stats 