"""
Model Training Script for Face Shape Classification
Based on the proven workflow from the example notebooks
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class FaceShapeTrainer:
    """
    Training pipeline for face shape classification
    Following the proven approach from the example notebooks
    """
    
    def __init__(self, data_dir: str = "backend/data/face_shapes"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.models_dir = self.data_dir / "models"
        
        # Face shape classes (matching example workflow)
        self.face_shapes = ['heart', 'long', 'oval', 'round', 'square']
        self.class_labels = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
        self.num_classes = len(self.face_shapes)
        
        # Training parameters
        self.img_size = (224, 224)
        self.batch_size = 32
        self.epochs = 50
        
        # Initialize model
        self.model = None
        self.history = None
        
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
    
    def load_data_from_directories(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data from organized directory structure"""
        images = []
        labels = []
        
        print("Loading data from directories...")
        
        for i, shape in enumerate(self.face_shapes):
            shape_dir = self.raw_dir / shape
            if not shape_dir.exists():
                print(f"Warning: Directory {shape_dir} does not exist")
                continue
            
            # Get all image files
            image_files = list(shape_dir.glob("*.jpg")) + list(shape_dir.glob("*.png"))
            
            print(f"Loading {len(image_files)} images for {shape} face shape")
            
            for img_path in image_files:
                try:
                    # Load and preprocess image using the exact workflow approach
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Extract face using MTCNN (same as in the notebooks)
                    face_img = self._extract_face_from_image(img)
                    
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    
                    # Normalize to [0,1]
                    img_rgb = img_rgb.astype(np.float32) / 255.0
                    
                    images.append(img_rgb)
                    labels.append(i)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Loaded {len(X)} total images")
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, self.num_classes)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, stratify=y, random_state=42
        )
        
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
    
    def create_model(self) -> Sequential:
        """Create VGG16-based model following the exact example workflow"""
        # Create base VGG16 model
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze ALL base layers as done in the notebook
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create model with the exact architecture from the notebook
        model = Sequential()
        model.add(base_model)
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model with Adam optimizer (as in notebook)
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model created successfully")
        self.model = model
        return model
    
    def train_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                   y_train: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train the model with callbacks and monitoring"""
        if self.model is None:
            self.create_model()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                str(self.models_dir / 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Setup data augmentation - exact parameters from notebook
        datagen = ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        datagen.fit(X_train)
        
        print("Starting training...")
        
        # Train model
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=self.epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        
        # Save final model
        final_model_path = self.models_dir / 'face_shape_classifier.h5'
        self.model.save(str(final_model_path))
        print(f"Model saved to {final_model_path}")
        
        return history.history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        if self.model is None:
            print("No model available for evaluation")
            return {}
        
        # Make predictions
        predictions = self.model.predict(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_labels,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_labels))
        
        return {
            'accuracy': accuracy,
            'classification_report': report
        }
    
    def run_complete_training(self) -> Dict:
        """Run the complete training pipeline"""
        print("="*50)
        print("FACE SHAPE CLASSIFICATION TRAINING")
        print("="*50)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data_from_directories()
        
        if len(X_train) == 0:
            print("Error: No training data found!")
            return {}
        
        # Create and train model
        self.create_model()
        training_history = self.train_model(X_train, X_test, y_train, y_test)
        
        # Evaluate model
        evaluation_results = self.evaluate_model(X_test, y_test)
        
        print("\nTraining completed successfully!")
        print(f"Final accuracy: {evaluation_results.get('accuracy', 0):.4f}")
        
        return evaluation_results

# Example usage
if __name__ == "__main__":
    trainer = FaceShapeTrainer()
    results = trainer.run_complete_training() 