#!/usr/bin/env python3
"""
Setup script for Real-Time Vehicle Detection and Traffic Flow Classification System
This script helps set up the project environment and downloads necessary files.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print(e.stdout)
        print(e.stderr)
        return False

def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return False
    
    pt_model = models_dir / "best.pt"
    onnx_model = models_dir / "best.onnx"
    
    if pt_model.exists():
        print("✅ PyTorch model (best.pt) found!")
    else:
        print("❌ PyTorch model (best.pt) not found!")
        
    if onnx_model.exists():
        print("✅ ONNX model (best.onnx) found!")
    else:
        print("❌ ONNX model (best.onnx) not found!")
    
    return pt_model.exists()

def create_data_directory():
    """Create data directory structure"""
    print("\n📁 Creating data directory structure...")
    
    data_dirs = [
        "Data",
        "Data/Vehicle_Detection_Image_Dataset",
        "Data/Vehicle_Detection_Image_Dataset/train",
        "Data/Vehicle_Detection_Image_Dataset/train/images",
        "Data/Vehicle_Detection_Image_Dataset/train/labels",
        "Data/Vehicle_Detection_Image_Dataset/valid",
        "Data/Vehicle_Detection_Image_Dataset/valid/images",
        "Data/Vehicle_Detection_Image_Dataset/valid/labels"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Data directory structure created!")

def check_sample_files():
    """Check for sample files"""
    print("\n🎥 Checking for sample files...")
    
    sample_video = Path("sample_video.mp4")
    dataset_sample_image = Path("Data/Vehicle_Detection_Image_Dataset/sample_image.jpg")
    dataset_sample_video = Path("Data/Vehicle_Detection_Image_Dataset/sample_video.mp4")
    
    missing_files = []
    
    if not sample_video.exists():
        missing_files.append("sample_video.mp4 (in root directory)")
    else:
        print("✅ sample_video.mp4 found!")
        
    if not dataset_sample_image.exists():
        missing_files.append("Data/Vehicle_Detection_Image_Dataset/sample_image.jpg")
    else:
        print("✅ Dataset sample image found!")
        
    if not dataset_sample_video.exists():
        missing_files.append("Data/Vehicle_Detection_Image_Dataset/sample_video.mp4")
    else:
        print("✅ Dataset sample video found!")
    
    if missing_files:
        print("\n⚠️  Missing sample files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nYou'll need to provide these files to run the complete project.")
    
    return len(missing_files) == 0

def create_sample_data_yaml():
    """Create a sample data.yaml file"""
    print("\n📝 Creating sample data.yaml file...")
    
    yaml_content = """# Vehicle Detection Dataset Configuration
path: Data/Vehicle_Detection_Image_Dataset  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images    # val images (relative to 'path')

# Classes
names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: bicycle

# Number of classes
nc: 5
"""
    
    yaml_path = Path("Data/Vehicle_Detection_Image_Dataset/data.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print("✅ Sample data.yaml created!")

def test_environment():
    """Test if the environment is working"""
    print("\n🧪 Testing environment...")
    
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not installed properly!")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully!")
    except ImportError:
        print("❌ Ultralytics not installed properly!")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available! Device: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available. Will use CPU.")
    except ImportError:
        print("❌ PyTorch not installed properly!")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up Real-Time Vehicle Detection and Traffic Flow Classification System")
    print("=" * 80)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Check model files
    if not check_model_files():
        print("\n⚠️  Model files are missing. The project may not work without them.")
    
    # Create data directory
    create_data_directory()
    
    # Create sample data.yaml
    create_sample_data_yaml()
    
    # Check sample files
    check_sample_files()
    
    # Test environment
    if test_environment():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your video files (sample_video.mp4) to test the system")
        print("2. Add dataset images to Data/Vehicle_Detection_Image_Dataset/ if you want to retrain")
        print("3. Run: python real_time_traffic_analysis_improved.py --help for usage options")
        print("4. For webcam: python real_time_traffic_analysis_improved.py --webcam")
    else:
        print("\n❌ Setup encountered issues. Please check the error messages above.")

if __name__ == "__main__":
    main()
