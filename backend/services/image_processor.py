import cv2
import numpy as np
from typing import Tuple, Optional, List
import io
from PIL import Image

# Try to import MTCNN for better face detection
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

class ImageProcessor:
    """
    Service for processing images and detecting faces
    Enhanced with MTCNN face detection from the example workflow
    """
    
    def __init__(self):
        # Initialize OpenCV face detector as fallback
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize MTCNN detector for better face detection
        if MTCNN_AVAILABLE:
            self.mtcnn_detector = MTCNN()
        else:
            self.mtcnn_detector = None
            print("MTCNN not available. Using OpenCV as fallback.")
        
    async def process_image(self, image_bytes: bytes) -> Tuple[Optional[List[List[float]]], Optional[np.ndarray]]:
        """
        Process uploaded image and extract face
        Updated to use MTCNN detection from the example workflow
        
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
            
            # Convert to OpenCV format (BGR)
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Try MTCNN detection first
            if self.mtcnn_detector:
                landmarks, face_region = self._detect_face_mtcnn(cv_image)
                
                # If MTCNN fails, use crop_and_resize as in the notebook
                if face_region is None:
                    face_region = self.crop_and_resize(cv_image)
                    landmarks = self._generate_simple_landmarks_from_bbox(0, 0, cv_image.shape[1], cv_image.shape[0])
            else:
                # Fallback to crop_and_resize if MTCNN not available
                face_region = self.crop_and_resize(cv_image)
                landmarks = self._generate_simple_landmarks_from_bbox(0, 0, cv_image.shape[1], cv_image.shape[0])
            
            if face_region is None:
                return None, None
            
            # Convert back to RGB for the classifier
            face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
            
            return landmarks, face_rgb
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None
    
    def _detect_face_mtcnn(self, cv_image: np.ndarray) -> Tuple[Optional[List[List[float]]], Optional[np.ndarray]]:
        """
        Detect face using MTCNN (more accurate)
        Based on the example workflow approach
        """
        try:
            # Convert BGR to RGB for MTCNN
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.mtcnn_detector.detect_faces(rgb_image)
            
            if not results:
                return None, None
            
            # Get the first (most confident) detection
            detection = results[0]
            x, y, width, height = detection['box']
            confidence = detection['confidence']
            
            # Only use detections with good confidence
            if confidence < 0.9:
                return None, None
            
            # Extract landmarks from MTCNN result
            landmarks = []
            if 'keypoints' in detection:
                keypoints = detection['keypoints']
                for key in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']:
                    if key in keypoints:
                        landmarks.append([float(keypoints[key][0]), float(keypoints[key][1])])
            
            # If no landmarks, create simple ones based on bounding box
            if not landmarks:
                landmarks = self._generate_simple_landmarks_from_bbox(x, y, width, height)
            
            # Extract face region with padding (following example workflow)
            face_region = self._extract_face_region(cv_image, x, y, width, height)
            
            return landmarks, face_region
            
        except Exception as e:
            print(f"Error in MTCNN detection: {e}")
            return None, None
    
    def _detect_face_opencv(self, cv_image: np.ndarray) -> Tuple[Optional[List[List[float]]], Optional[np.ndarray]]:
        """
        Detect face using OpenCV (fallback method)
        """
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                return None, None
            
            # Get the largest face
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = largest_face
            
            # Generate simple landmarks
            landmarks = self._generate_simple_landmarks_from_bbox(x, y, w, h)
            
            # Extract face region
            face_region = self._extract_face_region(cv_image, x, y, w, h)
            
            return landmarks, face_region
            
        except Exception as e:
            print(f"Error in OpenCV detection: {e}")
            return None, None
    
    def _extract_face_region(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Extract face region with proper padding and cropping
        Following the approach from the example notebooks
        """
        # Expand bounding box slightly
        adj_h = 10
        
        # Adjust y coordinates
        new_y1 = max(0, y - adj_h)
        new_y2 = min(image.shape[0], y + height + adj_h)
        new_height = new_y2 - new_y1
        
        # Make it square by adjusting width
        adj_w = int((new_height - width) / 2)
        new_x1 = max(0, x - adj_w)
        new_x2 = min(image.shape[1], x + width + adj_w)
        
        # Extract face region
        face_region = image[new_y1:new_y2, new_x1:new_x2]
        
        # Resize to 224x224 for the model
        face_resized = cv2.resize(face_region, (224, 224))
        
        return face_resized
    
    def _generate_simple_landmarks_from_bbox(self, x: int, y: int, w: int, h: int) -> List[List[float]]:
        """
        Generate simplified landmarks based on face bounding box
        For compatibility with existing code
        """
        landmarks = []
        
        # Key facial points based on bounding box
        # Left eye
        landmarks.append([float(x + w * 0.3), float(y + h * 0.4)])
        # Right eye  
        landmarks.append([float(x + w * 0.7), float(y + h * 0.4)])
        # Nose
        landmarks.append([float(x + w * 0.5), float(y + h * 0.55)])
        # Left mouth corner
        landmarks.append([float(x + w * 0.35), float(y + h * 0.7)])
        # Right mouth corner
        landmarks.append([float(x + w * 0.65), float(y + h * 0.7)])
        
        return landmarks
    
    def crop_and_resize(self, image: np.ndarray, target_w: int = 224, target_h: int = 224) -> np.ndarray:
        """
        Crop and resize image while maintaining aspect ratio
        From the example workflow
        """
        if image.ndim == 2:
            img_h, img_w = image.shape
        elif image.ndim == 3:
            img_h, img_w, channels = image.shape
        else:
            return cv2.resize(image, (target_w, target_h))
        
        target_aspect_ratio = target_w / target_h
        input_aspect_ratio = img_w / img_h
        
        if input_aspect_ratio > target_aspect_ratio:
            resize_w = int(input_aspect_ratio * target_h)
            resize_h = target_h
            img = cv2.resize(image, (resize_w, resize_h))
            crop_left = int((resize_w - target_w) / 2)
            crop_right = crop_left + target_w
            new_img = img[:, crop_left:crop_right]
        elif input_aspect_ratio < target_aspect_ratio:
            resize_w = target_w
            resize_h = int(target_w / input_aspect_ratio)
            img = cv2.resize(image, (resize_w, resize_h))
            crop_top = int((resize_h - target_h) / 4)
            crop_bottom = crop_top + target_h
            new_img = img[crop_top:crop_bottom, :]
        else:
            new_img = cv2.resize(image, (target_w, target_h))
        
        return new_img 