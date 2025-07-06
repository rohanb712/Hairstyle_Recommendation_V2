import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image
import os
from pathlib import Path

# Try to import MTCNN - install with: pip install mtcnn
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("MTCNN not available. Install with: pip install mtcnn")

class VGGFaceShapeClassifier(nn.Module):
    """
    Face shape classifier using VGG16 transfer learning approach
    Based on the proven workflow from the example notebooks, now using PyTorch
    """
    
    def __init__(self, num_classes=5):
        super(VGGFaceShapeClassifier, self).__init__()
        
        # Load pretrained VGG16
        self.vgg16 = models.vgg16(pretrained=True)
        
        # Freeze all layers except the last few
        for param in self.vgg16.parameters():
            param.requires_grad = False
        
        # Unfreeze the last 4 layers for fine-tuning
        for param in self.vgg16.classifier[-4:].parameters():
            param.requires_grad = True
        
        # Replace the classifier to match our architecture
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize the new layers
        for module in self.vgg16.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.vgg16(x)

class VGGFaceShapeClassifierWrapper:
    """
    Wrapper class for the PyTorch VGG face shape classifier
    Maintains the same interface as the original TensorFlow version
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = VGGFaceShapeClassifier(num_classes=5).to(self.device)
        self.face_detector = None
        
        # Face shape labels matching the example workflow
        self.face_shapes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        self.label_dict = {i: shape for i, shape in enumerate(self.face_shapes)}
        
        # Initialize MTCNN face detector
        if MTCNN_AVAILABLE:
            self.face_detector = MTCNN()
        
        # Image preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print("VGG16 face shape classifier created (PyTorch)")
    
    def load_model(self, model_path: str):
        """Load a saved model"""
        try:
            print(f"🔄 Loading model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Get model info if available
            if 'face_shapes' in checkpoint:
                self.face_shapes = checkpoint['face_shapes']
                self.label_dict = {i: shape for i, shape in enumerate(self.face_shapes)}
            
            print(f"✅ Trained VGG16 model loaded successfully!")
            print(f"📋 Face shapes: {self.face_shapes}")
            print(f"🔧 Device: {self.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Using untrained model (will give poor results)")
    
    def save_model(self, model_path: str):
        """Save the current model"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'face_shapes': self.face_shapes,
                'label_dict': self.label_dict
            }, model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
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
            # Extract and preprocess face
            face_extracted = self.extract_face(face_image)
            
            # Convert to RGB if needed
            if len(face_extracted.shape) == 3 and face_extracted.shape[2] == 3:
                # Already RGB
                processed_face = face_extracted
            else:
                # Convert BGR to RGB
                processed_face = cv2.cvtColor(face_extracted, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            input_tensor = self.transform(processed_face).unsqueeze(0).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get predicted class and confidence
                confidence, predicted_class = torch.max(probabilities, 1)
                
                face_shape = self.label_dict[predicted_class.item()]
                confidence_score = confidence.item()
            
            return face_shape, confidence_score
            
        except Exception as e:
            print(f"Error classifying face: {e}")
            return "Oval", 0.5  # Default fallback


class FaceClassifier:
    """
    Wrapper class to maintain compatibility with existing code
    while using the new PyTorch-based classifier
    """
    
    def __init__(self):
        # Initialize the PyTorch-based classifier with trained model
        models_dir = Path("backend/data/face_shapes/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = models_dir / "face_shape_classifier.pth"
        
        # Check if trained model exists
        if model_path.exists():
            print(f"🎯 Loading trained VGG16 model from: {model_path}")
            print(f"📊 Model file size: {model_path.stat().st_size / (1024*1024):.1f} MB")
            self.classifier = VGGFaceShapeClassifierWrapper(model_path=str(model_path))
        else:
            print(f"❌ Trained model not found at: {model_path}")
            print("🔧 Please train the model first using: python backend/train_model.py")
            print("⚠️  Using untrained model for now (will give poor results)")
            self.classifier = VGGFaceShapeClassifierWrapper(model_path=None)
        
        # Map new labels to original format for compatibility
        self.shape_mapping = {
            'Heart': 'heart',
            'Oblong': 'oblong',  # Keep oblong as oblong
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
            model_path = models_dir / "face_shape_classifier.pth"
        
        self.classifier.save_model(str(model_path))
    
    def get_training_ready_classifier(self):
        """Get the underlying PyTorch classifier for training"""
        return self.classifier 