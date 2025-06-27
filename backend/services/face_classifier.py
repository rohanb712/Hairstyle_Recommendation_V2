import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
import os
from pathlib import Path

# Try to import MTCNN - install with: pip install mtcnn
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("MTCNN not available. Install with: pip install mtcnn")

class VGGFaceShapeClassifier:
    """
    Face shape classifier using VGG16 transfer learning approach
    Based on the proven workflow from the example notebooks
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.face_detector = None
        
        # Face shape labels matching the example workflow
        self.face_shapes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        self.label_dict = {i: shape for i, shape in enumerate(self.face_shapes)}
        
        # Initialize MTCNN face detector
        if MTCNN_AVAILABLE:
            self.face_detector = MTCNN()
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.create_model()
    
    def create_model(self):
        """
        Create VGG16-based model with transfer learning
        Following the architecture from the example notebooks
        """
        # Create base VGG16 model
        base_model = VGG16(
            weights='imagenet',  # Use ImageNet weights initially
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze base layers
        for layer in base_model.layers[:-4]:  # Freeze all but last 4 layers
            layer.trainable = False
        
        # Create model
        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(len(self.face_shapes), activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("VGG16 face shape classifier created")
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        try:
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.create_model()
    
    def save_model(self, model_path: str):
        """Save the current model"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
    
    def crop_and_resize(self, image: np.ndarray, target_w: int = 224, target_h: int = 224) -> np.ndarray:
        """
        Crop and resize image to target size while maintaining aspect ratio
        Based on the function from the example notebooks
        """
        if image.ndim == 2:
            img_h, img_w = image.shape
        elif image.ndim == 3:
            img_h, img_w, channels = image.shape
        
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
    
    def extract_face(self, img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Extract face from image using MTCNN face detection
        Based on the function from the example notebooks
        """
        if not self.face_detector:
            # Fallback to crop and resize if MTCNN not available
            return self.crop_and_resize(img, target_size[0], target_size[1])
        
        try:
            # Detect faces
            results = self.face_detector.detect_faces(img)
            
            if not results:
                # If no face detected, use crop and resize
                return self.crop_and_resize(img, target_size[0], target_size[1])
            
            # Get the first (most confident) face detection
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height
            
            # Expand bounding box slightly
            adj_h = 10
            
            # Adjust y coordinates
            new_y1 = max(0, y1 - adj_h)
            new_y2 = min(img.shape[0], y1 + height + adj_h)
            new_height = new_y2 - new_y1
            
            # Make it square by adjusting width
            adj_w = int((new_height - width) / 2)
            new_x1 = max(0, x1 - adj_w)
            new_x2 = min(img.shape[1], x2 + adj_w)
            
            # Extract face region
            face_region = img[new_y1:new_y2, new_x1:new_x2]
            
            # Resize to target size
            face_resized = cv2.resize(face_region, target_size)
            
            return face_resized
            
        except Exception as e:
            print(f"Error in face extraction: {e}")
            return self.crop_and_resize(img, target_size[0], target_size[1])
    
    async def classify_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify face shape from image
        
        Args:
            face_image: Input face image as numpy array (RGB)
            
        Returns:
            Tuple of (face_shape, confidence)
        """
        try:
            if self.model is None:
                return "Oval", 0.5  # Default fallback
            
            # Extract and preprocess face
            face_extracted = self.extract_face(face_image)
            
            # Convert to RGB if needed
            if len(face_extracted.shape) == 3 and face_extracted.shape[2] == 3:
                # Already RGB
                processed_face = face_extracted
            else:
                # Convert BGR to RGB
                processed_face = cv2.cvtColor(face_extracted, cv2.COLOR_BGR2RGB)
            
            # Normalize to 0-1 range
            processed_face = processed_face.astype(np.float32) / 255.0
            
            # Reshape for model input
            model_input = np.expand_dims(processed_face, axis=0)
            
            # Make prediction
            predictions = self.model.predict(model_input, verbose=0)
            
            # Get predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            face_shape = self.label_dict[predicted_class]
            
            return face_shape, confidence
            
        except Exception as e:
            print(f"Error classifying face: {e}")
            return "Oval", 0.5  # Default fallback


class FaceClassifier:
    """
    Wrapper class to maintain compatibility with existing code
    while using the new VGG-based classifier
    """
    
    def __init__(self):
        # Initialize the VGG-based classifier
        models_dir = Path("backend/data/face_shapes/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "face_shape_classifier.h5"
        self.classifier = VGGFaceShapeClassifier(
            model_path=str(model_path) if model_path.exists() else None
        )
        
        # Map new labels to original format for compatibility
        self.shape_mapping = {
            'Heart': 'heart',
            'Oblong': 'long',  # Map oblong to long
            'Oval': 'oval',
            'Round': 'round',
            'Square': 'square'
        }
    
    async def classify_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify face shape - compatible with existing interface
        
        Args:
            face_image: Aligned face image as numpy array (RGB)
            
        Returns:
            Tuple of (face_shape, confidence) in lowercase format
        """
        try:
            shape, confidence = await self.classifier.classify_face(face_image)
            
            # Map to lowercase format expected by existing code
            mapped_shape = self.shape_mapping.get(shape, 'oval').lower()
            
            return mapped_shape, confidence
                
        except Exception as e:
            print(f"Error in face classification: {e}")
            return "oval", 0.5
    
    def save_model(self, model_path: str = None):
        """Save the trained model"""
        if model_path is None:
            models_dir = Path("backend/data/face_shapes/models")
            model_path = models_dir / "face_shape_classifier.h5"
        
        self.classifier.save_model(str(model_path))
    
    def get_training_ready_classifier(self):
        """Get the underlying VGG classifier for training"""
        return self.classifier 