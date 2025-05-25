import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple
from PIL import Image

class FaceShapeClassifier(nn.Module):
    """Simple CNN for face shape classification"""
    
    def __init__(self, num_classes=5):
        super(FaceShapeClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FaceClassifier:
    """Service for classifying face shapes"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Face shape labels
        self.face_shapes = ['Oval', 'Round', 'Square', 'Heart', 'Long']
        
        # Initialize model (for V1, we'll use a simple rule-based approach)
        self._init_model()
    
    def _init_model(self):
        """Initialize the face shape classification model"""
        # For V1, we'll create a dummy model since we don't have trained weights
        # In production, you would load a pre-trained model
        self.model = FaceShapeClassifier(num_classes=len(self.face_shapes))
        self.model.to(self.device)
        self.model.eval()
        
        # For demo purposes, we'll use a simple rule-based classifier
        # In production, load actual trained weights:
        # self.model.load_state_dict(torch.load('path/to/model.pth'))
    
    async def classify_face(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify face shape from aligned face image
        
        Args:
            face_image: Aligned face image as numpy array (RGB)
            
        Returns:
            Tuple of (face_shape, confidence)
        """
        try:
            # For V1, we'll use a simple rule-based approach based on face dimensions
            # In production, use the actual trained model
            
            height, width = face_image.shape[:2]
            aspect_ratio = height / width
            
            # Simple rule-based classification for demo
            if aspect_ratio > 1.3:
                face_shape = "Long"
                confidence = 0.85
            elif aspect_ratio < 0.9:
                face_shape = "Round"
                confidence = 0.80
            elif 0.9 <= aspect_ratio <= 1.1:
                face_shape = "Square"
                confidence = 0.75
            elif 1.1 < aspect_ratio <= 1.2:
                face_shape = "Heart"
                confidence = 0.70
            else:
                face_shape = "Oval"
                confidence = 0.90
            
            return face_shape, confidence
            
        except Exception as e:
            print(f"Error classifying face: {e}")
            # Return default classification
            return "Oval", 0.5
    
    def _classify_with_model(self, face_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify using the actual PyTorch model (for future use)
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(face_image)
            
            # Apply transforms
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                
                face_shape = self.face_shapes[predicted.item()]
                confidence_score = confidence.item()
                
                return face_shape, confidence_score
                
        except Exception as e:
            print(f"Error in model classification: {e}")
            return "Oval", 0.5 