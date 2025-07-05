"""
Model Training Script for Face Shape Classification
Based on the proven workflow from the example notebooks, now using PyTorch
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Import our PyTorch classifier
from .face_classifier import VGGFaceShapeClassifier

class FaceShapeDataset(Dataset):
    """Custom dataset for face shape classification"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

class FaceShapeTrainer:
    """
    Training pipeline for face shape classification
    Following the proven approach from the example notebooks, now using PyTorch
    """
    
    def __init__(self, data_dir: str = "backend/data/face_shapes"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.models_dir = self.data_dir / "models"
        
        # Face shape classes (matching example workflow)
        self.face_shapes = ['heart', 'oblong', 'oval', 'round', 'square']
        self.class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        self.num_classes = len(self.face_shapes)
        
        # Training parameters
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = None
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Initialize MTCNN for face detection (matching the notebooks)
        try:
            from mtcnn import MTCNN
            self.mtcnn_detector = MTCNN()
            print("MTCNN face detector initialized")
        except ImportError:
            print("MTCNN not available. Will use crop_and_resize fallback.")
            self.mtcnn_detector = None
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_data_from_directories(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data from organized directory structure (train/test split)"""
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        
        print("Loading data from train/test directories...")
        
        # Load training data
        train_dir = self.raw_dir / "train"
        if train_dir.exists():
            print("Loading training data...")
            for i, shape in enumerate(self.face_shapes):
                shape_dir = train_dir / shape
                if not shape_dir.exists():
                    print(f"Warning: Directory {shape_dir} does not exist")
                    continue
                
                # Get all image files
                image_files = list(shape_dir.glob("*.jpg")) + list(shape_dir.glob("*.png"))
                
                print(f"Loading {len(image_files)} training images for {shape} face shape")
                
                for img_path in image_files:
                    try:
                        # Handle Unicode filenames by using numpy and cv2 workaround
                        img_path_str = str(img_path)
                        
                        # Try to read the image
                        img = cv2.imread(img_path_str)
                        if img is None:
                            # Try alternative method for Unicode filenames
                            try:
                                img_array = np.fromfile(img_path_str, dtype=np.uint8)
                                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            except:
                                continue
                        
                        if img is None:
                            continue
                        
                        # Extract face using MTCNN (same as in the notebooks)
                        face_img = self._extract_face_from_image(img)
                        
                        # Convert BGR to RGB
                        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        # Store as uint8 for transforms
                        train_images.append(img_rgb)
                        train_labels.append(i)
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
        
        # Load test data
        test_dir = self.raw_dir / "test"
        if test_dir.exists():
            print("Loading test data...")
            for i, shape in enumerate(self.face_shapes):
                shape_dir = test_dir / shape
                if not shape_dir.exists():
                    print(f"Warning: Directory {shape_dir} does not exist")
                    continue
                
                # Get all image files
                image_files = list(shape_dir.glob("*.jpg")) + list(shape_dir.glob("*.png"))
                
                print(f"Loading {len(image_files)} test images for {shape} face shape")
                
                for img_path in image_files:
                    try:
                        # Handle Unicode filenames by using numpy and cv2 workaround
                        img_path_str = str(img_path)
                        
                        # Try to read the image
                        img = cv2.imread(img_path_str)
                        if img is None:
                            # Try alternative method for Unicode filenames
                            try:
                                img_array = np.fromfile(img_path_str, dtype=np.uint8)
                                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                            except:
                                continue
                        
                        if img is None:
                            continue
                        
                        # Extract face using MTCNN (same as in the notebooks)
                        face_img = self._extract_face_from_image(img)
                        
                        # Convert BGR to RGB
                        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                        
                        # Store as uint8 for transforms
                        test_images.append(img_rgb)
                        test_labels.append(i)
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
        
        # Convert to numpy arrays
        X_train = np.array(train_images)
        y_train = np.array(train_labels)
        X_test = np.array(test_images)
        y_test = np.array(test_labels)
        
        print(f"Loaded {len(X_train)} training images and {len(X_test)} test images")
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        
        return X_train, X_test, y_train, y_test
    
    def _extract_face_from_image(self, img: np.ndarray) -> np.ndarray:
        """
        Extract face from image using MTCNN, fallback to crop_and_resize
        Follows the exact approach from the example notebooks
        """
        try:
            # Try MTCNN detection first
            if hasattr(self, 'mtcnn_detector') and self.mtcnn_detector:
                results = self.mtcnn_detector.detect_faces(img)
                
                if results:
                    x1, y1, width, height = results[0]['box']
                    
                    # Expand bounding box (from notebooks)
                    adj_h = 10
                    new_y1 = max(0, y1 - adj_h)
                    new_y2 = min(img.shape[0], y1 + height + adj_h)
                    new_height = new_y2 - new_y1
                    
                    # Make square
                    adj_w = int((new_height - width) / 2)
                    new_x1 = max(0, x1 - adj_w)
                    new_x2 = min(img.shape[1], x1 + width + adj_w)
                    
                    # Extract and resize
                    face_region = img[new_y1:new_y2, new_x1:new_x2]
                    return cv2.resize(face_region, (224, 224))
            
            # Fallback to crop_and_resize
            return self._crop_and_resize(img)
            
        except Exception as e:
            print(f"Face extraction error: {e}")
            return self._crop_and_resize(img)
    
    def _crop_and_resize(self, image: np.ndarray, target_w: int = 224, target_h: int = 224) -> np.ndarray:
        """Crop and resize maintaining aspect ratio (from notebooks)"""
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
    
    def create_model(self) -> VGGFaceShapeClassifier:
        """Create VGG16-based model following the exact example workflow"""
        model = VGGFaceShapeClassifier(num_classes=self.num_classes)
        model = model.to(self.device)
        
        print("VGG16-based model created (PyTorch)")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        return model
    
    def train_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Train the model using the exact training approach from the notebooks
        """
        print("Creating model...")
        self.model = self.create_model()
        
        # Create datasets
        train_dataset = FaceShapeDataset(X_train, y_train, transform=self.train_transform)
        val_dataset = FaceShapeDataset(X_test, y_test, transform=self.val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        early_stopping_patience = 10
        
        print(f"Starting training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}] - '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                model_path = self.models_dir / "face_shape_classifier.pth"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                    'face_shapes': self.face_shapes,
                    'class_labels': self.class_labels
                }, model_path)
                
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        
        return {
            'best_val_accuracy': best_val_acc / 100.0,
            'final_train_accuracy': train_acc / 100.0,
            'epochs_trained': epoch + 1,
            'history': self.history
        }
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the trained model on test data
        """
        if self.model is None:
            print("No model to evaluate!")
            return {}
        
        # Create test dataset and loader
        test_dataset = FaceShapeDataset(X_test, y_test, transform=self.val_transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Evaluation
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate accuracy
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        
        # Generate classification report
        report = classification_report(
            all_labels, all_predictions, 
            target_names=self.class_labels, 
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, target_names=self.class_labels))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'true_labels': all_labels
        }
    
    def run_complete_training(self) -> Dict:
        """
        Run the complete training pipeline
        """
        print("="*60)
        print("STARTING COMPLETE TRAINING PIPELINE")
        print("="*60)
        
        try:
            # Load data
            X_train, X_test, y_train, y_test = self.load_data_from_directories()
            
            if len(X_train) == 0:
                print("No training data found!")
                return {}
            
            # Train model
            training_results = self.train_model(X_train, X_test, y_train, y_test)
            
            # Evaluate model
            evaluation_results = self.evaluate_model(X_test, y_test)
            
            # Combine results
            results = {**training_results, **evaluation_results}
            
            # Save training history
            history_path = self.models_dir / "training_history.json"
            with open(history_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                history_json = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        history_json[key] = value.tolist()
                    else:
                        history_json[key] = value
                
                json.dump(history_json, f, indent=2)
            
            print(f"Training results saved to {history_path}")
            
            return results
            
        except Exception as e:
            print(f"Training pipeline error: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    trainer = FaceShapeTrainer()
    results = trainer.run_complete_training() 