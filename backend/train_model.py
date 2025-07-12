#!/usr/bin/env python3
"""
Face Shape Classification Model Training Script

This script trains the VGG16-based face shape classifier using your mixed-gender dataset.
Make sure your images are organized in the proper directory structure before running.

Usage:
    python train_model.py

Requirements:
    - Dataset organized in backend/data/face_shapes/raw/{shape}/
    - At least 200+ images per face shape for reasonable performance
    - MTCNN installed for face detection during training
    - PyTorch and torchvision installed
"""

import sys
import os
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.append(str(backend_dir))

from services.model_trainer import FaceShapeTrainer

def check_dataset_structure(data_dir: Path):
    """Check if the dataset is properly organized"""
    raw_dir = data_dir / "raw"
    required_shapes = ['heart', 'oblong', 'oval', 'round', 'square']
    
    print("Checking dataset structure...")
    
    if not raw_dir.exists():
        print(f"Error: {raw_dir} directory not found!")
        print("Please create the directory structure and add your images:")
        for shape in required_shapes:
            print(f"  backend/data/face_shapes/raw/train/{shape}/")
            print(f"  backend/data/face_shapes/raw/test/{shape}/")
        return False
    
    # Check for train/test directory structure
    train_dir = raw_dir / "train"
    test_dir = raw_dir / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        print(f"Error: Expected train/test directory structure not found!")
        print("Please organize your data as:")
        for shape in required_shapes:
            print(f"  backend/data/face_shapes/raw/train/{shape}/")
            print(f"  backend/data/face_shapes/raw/test/{shape}/")
        return False
    
    total_images = 0
    train_images = 0
    test_images = 0
    
    print("\nTraining data:")
    for shape in required_shapes:
        shape_dir = train_dir / shape
        if not shape_dir.exists():
            print(f"Warning: {shape_dir} directory not found!")
            continue
        
        # Count image files
        image_files = list(shape_dir.glob("*.jpg")) + list(shape_dir.glob("*.png"))
        image_count = len(image_files)
        train_images += image_count
        
        status = "GOOD" if image_count >= 100 else "OK" if image_count >= 50 else "LOW"
        print(f"{status} {shape}: {image_count} images")
        
        if image_count < 50:
            print(f"  Recommendation: Add more images for better performance")
    
    print("\nTest data:")
    for shape in required_shapes:
        shape_dir = test_dir / shape
        if not shape_dir.exists():
            print(f"Warning: {shape_dir} directory not found!")
            continue
        
        # Count image files
        image_files = list(shape_dir.glob("*.jpg")) + list(shape_dir.glob("*.png"))
        image_count = len(image_files)
        test_images += image_count
        
        status = "GOOD" if image_count >= 20 else "OK" if image_count >= 10 else "LOW"
        print(f"{status} {shape}: {image_count} images")
    
    total_images = train_images + test_images
    print(f"\nTotal images: {total_images} (Train: {train_images}, Test: {test_images})")
    
    if total_images < 250:
        print("Error: Not enough training data!")
        print("Recommendation: At least 50 images per face shape (250 total minimum)")
        return False
    
    return True

def main():
    print("Face Shape Classification Model Training (PyTorch)")
    print("="*50)
    
    # Check dataset
    data_dir = Path("backend/data/face_shapes")
    if not check_dataset_structure(data_dir):
        return
    
    print("\nDataset structure looks good!")
    
    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = FaceShapeTrainer()
    
    # Check MTCNN availability
    if trainer.mtcnn_detector is None:
        print("Warning: MTCNN not available. Install with: pip install mtcnn")
        print("Will use crop_and_resize fallback for preprocessing")
    else:
        print("MTCNN face detector ready")
    
    # Ask for confirmation
    print(f"\nTraining Configuration:")
    print(f"  Framework: PyTorch")
    print(f"  Device: {trainer.device}")
    print(f"  Input size: 224×224 RGB images")
    print(f"  Architecture: VGG16 transfer learning") 
    print(f"  Face shapes: Heart, Oblong, Oval, Round, Square")
    print(f"  Data augmentation: rotation ±20°, horizontal flip")
    print(f"  Expected accuracy: 85-92% (based on proven workflow)")
    print(f"  Training time: ~30-60 minutes (depending on dataset size)")
    
    response = input("\nStart training? (y/N): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        return
    
    # Run training
    print("\nStarting training pipeline...")
    try:
        results = trainer.run_complete_training()
        
        if results and 'accuracy' in results:
            accuracy = results['accuracy']
            print(f"\nTraining completed successfully!")
            print(f"Final test accuracy: {accuracy:.1%}")
            
            if accuracy >= 0.85:
                print("Excellent performance! Model is ready for production.")
            elif accuracy >= 0.75:
                print("Good performance! Model should work well.")
            else:
                print("Lower than expected accuracy. Consider:")
                print("  Adding more training data")
                print("  Checking image quality")
                print("  Ensuring proper face visibility in images")
            
            print(f"\nModel saved to: backend/data/face_shapes/models/face_shape_classifier.pth")
            print("The system will now automatically use this trained model!")
            
        else:
            print("Training failed. Check the error messages above.")
            
    except Exception as e:
        print(f"Training error: {e}")
        print("Please check your dataset and try again.")

if __name__ == "__main__":
    main() 