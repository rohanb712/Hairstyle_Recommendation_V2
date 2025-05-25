import cv2
import numpy as np
from typing import Tuple, Optional, List
import io
from PIL import Image

class ImageProcessor:
    """Service for processing images and detecting facial landmarks"""
    
    def __init__(self):
        # Initialize OpenCV face detector and landmark predictor
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # For landmark detection, we'll use a simple approach for V1
        # In production, you'd want to use dlib or MediaPipe for better accuracy
        self.landmark_detector = None
        
    async def process_image(self, image_bytes: bytes) -> Tuple[Optional[List[List[float]]], Optional[np.ndarray]]:
        """
        Process uploaded image and detect facial landmarks
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (landmarks, aligned_face_image)
        """
        try:
            # Convert bytes to PIL Image
            pil_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None, None
            
            # Get the largest face (assuming it's the main subject)
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = cv_image[y:y+h, x:x+w]
            
            # For V1, we'll create simple landmark points based on face detection
            # In production, use proper landmark detection like dlib or MediaPipe
            landmarks = self._generate_simple_landmarks(x, y, w, h)
            
            # Align and resize face for classification
            aligned_face = self._align_face(face_roi)
            
            return landmarks, aligned_face
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None
    
    def _generate_simple_landmarks(self, x: int, y: int, w: int, h: int) -> List[List[float]]:
        """
        Generate simple landmark points based on face bounding box
        This is a simplified approach for V1 - in production use proper landmark detection
        """
        landmarks = []
        
        # Generate 68 landmark points (standard format)
        # This is a simplified approximation based on face rectangle
        
        # Jaw line (17 points)
        for i in range(17):
            lx = x + (i / 16.0) * w
            ly = y + h * 0.8  # Bottom of face
            landmarks.append([float(lx), float(ly)])
        
        # Right eyebrow (5 points)
        for i in range(5):
            lx = x + w * (0.2 + i * 0.15)
            ly = y + h * 0.3
            landmarks.append([float(lx), float(ly)])
        
        # Left eyebrow (5 points)
        for i in range(5):
            lx = x + w * (0.55 + i * 0.15)
            ly = y + h * 0.3
            landmarks.append([float(lx), float(ly)])
        
        # Nose (9 points)
        for i in range(9):
            lx = x + w * 0.5
            ly = y + h * (0.4 + i * 0.05)
            landmarks.append([float(lx), float(ly)])
        
        # Right eye (6 points)
        for i in range(6):
            lx = x + w * (0.25 + i * 0.05)
            ly = y + h * 0.4
            landmarks.append([float(lx), float(ly)])
        
        # Left eye (6 points)
        for i in range(6):
            lx = x + w * (0.65 + i * 0.05)
            ly = y + h * 0.4
            landmarks.append([float(lx), float(ly)])
        
        # Mouth (20 points)
        for i in range(20):
            lx = x + w * (0.3 + (i / 19.0) * 0.4)
            ly = y + h * 0.65
            landmarks.append([float(lx), float(ly)])
        
        return landmarks
    
    def _align_face(self, face_image: np.ndarray) -> np.ndarray:
        """
        Align and resize face image for classification
        """
        # Resize to standard size for face classification
        aligned_face = cv2.resize(face_image, (224, 224))
        
        # Convert to RGB for PyTorch model
        aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        
        return aligned_face 