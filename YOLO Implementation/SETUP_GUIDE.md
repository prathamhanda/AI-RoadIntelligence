# 🚗 Real-Time Vehicle Detection Setup Guide

This guide will help you set up and run the Real-Time Vehicle Detection and Traffic Flow Classification System on your local machine.

## 📋 Prerequisites

- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **Git** (for cloning repositories)
- **Webcam** (optional, for live detection)

## 🚀 Quick Setup

### 1. Run the Setup Script
```bash
python setup.py
```

This will automatically:
- Check your Python version
- Install all required packages
- Create necessary directory structure
- Verify your environment

### 2. Manual Setup (Alternative)

If the setup script doesn't work, follow these manual steps:

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Create Data Directories
```bash
mkdir -p Data/Vehicle_Detection_Image_Dataset/train/images
mkdir -p Data/Vehicle_Detection_Image_Dataset/train/labels
mkdir -p Data/Vehicle_Detection_Image_Dataset/valid/images
mkdir -p Data/Vehicle_Detection_Image_Dataset/valid/labels
```

## 📁 Required Files

To run the complete system, you need these files:

### ✅ Already Included
- `models/best.pt` - Trained YOLOv8 model (PyTorch format)
- `models/best.onnx` - Trained YOLOv8 model (ONNX format)
- Training and validation result images in `images/`

### ❗ Files You Need to Provide

#### 1. Sample Video for Testing
- **File**: `sample_video.mp4`
- **Location**: Root directory
- **Purpose**: Test the real-time detection system
- **Where to get**: 
  - Record your own traffic video
  - Download from traffic video datasets
  - Use any MP4 video with vehicles

#### 2. Dataset Files (Optional - for retraining)
If you want to retrain the model or run the complete notebook:

- `Data/Vehicle_Detection_Image_Dataset/sample_image.jpg` - Test image
- `Data/Vehicle_Detection_Image_Dataset/sample_video.mp4` - Test video
- Training images in `Data/Vehicle_Detection_Image_Dataset/train/images/`
- Validation images in `Data/Vehicle_Detection_Image_Dataset/valid/images/`
- Corresponding label files in respective `labels/` folders

## 🎯 How to Run

### Option 1: Real-time Detection with Video File
```bash
python real_time_traffic_analysis_improved.py --source your_video.mp4
```

### Option 2: Real-time Detection with Webcam
```bash
python real_time_traffic_analysis_improved.py --webcam
```

### Option 3: Original Script (Basic)
```bash
python real_time_traffic_analysis.py
```

### Option 4: Jupyter Notebook (Complete Training Pipeline)
```bash
jupyter notebook Notebook.ipynb
```

## ⚙️ Command Line Options

The improved script supports various options:

```bash
python real_time_traffic_analysis_improved.py [OPTIONS]

Options:
  --source PATH       Input video file path (default: sample_video.mp4)
  --weights PATH      Model weights path (default: models/best.pt)
  --output PATH       Output video path (default: processed_sample_video.avi)
  --conf FLOAT        Confidence threshold (default: 0.4)
  --webcam           Use webcam instead of video file
  --help             Show help message
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. "ModuleNotFoundError: No module named 'ultralytics'"
```bash
pip install ultralytics
```

#### 2. "Error: Video file not found!"
- Ensure your video file exists in the specified path
- Use absolute path if relative path doesn't work
- Try using `--webcam` flag for live detection

#### 3. "Error: Model weights file not found!"
- Ensure `models/best.pt` exists
- Download from the original repository if missing

#### 4. CUDA Issues
- The system works on both CPU and GPU
- For GPU acceleration, ensure CUDA-compatible PyTorch is installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 5. OpenCV Issues
```bash
pip uninstall opencv-python opencv-python-headless
pip install opencv-python
```

### Performance Tips

1. **For better performance**: Use GPU-enabled PyTorch
2. **For lower latency**: Reduce confidence threshold (--conf 0.3)
3. **For better accuracy**: Increase confidence threshold (--conf 0.6)

## 📊 Expected Output

The system will:
1. **Detect vehicles** in each frame
2. **Count vehicles** in left and right lanes
3. **Classify traffic intensity** as "Smooth" or "Heavy"
4. **Display real-time results** with bounding boxes and annotations
5. **Save processed video** (when using video files)

### Controls
- **Press 'q'** to quit the application
- **Close window** to stop processing

## 🎮 Testing the System

### Quick Test with Webcam
```bash
python real_time_traffic_analysis_improved.py --webcam
```

### Test with Sample Video
1. Add any MP4 video file as `sample_video.mp4`
2. Run:
```bash
python real_time_traffic_analysis_improved.py
```

## 📝 Next Steps

1. **Customize lane boundaries**: Edit the `vertices1` and `vertices2` coordinates in the script
2. **Adjust detection parameters**: Modify confidence thresholds and traffic intensity thresholds
3. **Add new features**: Extend the script for speed detection, vehicle counting, etc.
4. **Retrain the model**: Use the Jupyter notebook with your own dataset

## 🆘 Getting Help

If you encounter issues:

1. **Check the console output** for error messages
2. **Verify all files are in place** using the setup script
3. **Test with webcam first** to ensure the environment works
4. **Check Python and package versions**

## 📋 System Requirements

- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: Optional (NVIDIA GPU with CUDA support for better performance)
- **Storage**: 2GB free space
- **OS**: Windows 10/11, macOS, or Linux

---

🎉 **You're all set!** The system should now be ready to detect vehicles and analyze traffic flow in real-time.
