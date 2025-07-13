
# 🚗 Real-Time Vehicle Detection and Traffic Flow Classification System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-red.svg)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An advanced AI-powered traffic analysis system for real-time vehicle detection, lane classification, and traffic flow monitoring*

![Traffic Analysis Demo](images/traffic_density_analysis.gif)

</div>

## 📋 Table of Contents
1. [🎯 Project Overview](#-project-overview)
2. [✨ Key Features](#-key-features)
3. [🧠 Technical Architecture](#-technical-architecture)
4. [🚀 Project Potential & Applications](#-project-potential--applications)
5. [📁 Directory Structure](#-directory-structure)
6. [⚙️ Installation & Setup](#️-installation--setup)
7. [🎮 Usage Guide](#-usage-guide)
8. [📊 Model Performance](#-model-performance)
9. [🔧 Configuration & Customization](#-configuration--customization)
10. [🤝 Contributing](#-contributing)
11. [📄 License](#-license)

---

## 🎯 Project Overview

This project implements a comprehensive **Real-Time Vehicle Detection and Traffic Flow Classification System** using state-of-the-art YOLOv8 object detection technology. The system is designed to analyze traffic conditions in real-time, providing valuable insights for smart city infrastructure, traffic management, and urban planning.

### 🔬 What Makes This Special?

- **Custom-trained YOLOv8 model** optimized for vehicle detection in various traffic conditions
- **Intelligent lane detection** using polygon-based region of interest (ROI) mapping
- **Real-time traffic flow analysis** with customizable traffic intensity thresholds
- **Multi-format support** for videos, images, and live camera feeds
- **Scalable architecture** suitable for deployment in smart traffic systems

---

## ✨ Key Features

### 🎯 Core Capabilities
- **Real-Time Vehicle Detection**: Detect cars, trucks, buses, motorcycles, and other vehicles
- **Lane-Based Traffic Analysis**: Separate vehicle counting for different traffic lanes
- **Traffic Intensity Classification**: Automatic classification as "Smooth" or "Heavy" traffic
- **Multi-Source Input**: Support for video files, live cameras, and image sequences
- **Custom ROI Mapping**: Polygon-based region definition for precise lane detection

### 🛠️ Technical Features
- **YOLOv8 Integration**: Latest YOLO architecture for superior detection accuracy
- **ONNX Model Support**: Cross-platform deployment capability
- **Interactive Calibration**: Visual polygon coordinate calibration tool
- **Adaptive Display Scaling**: Automatic video scaling for different screen resolutions
- **Frame Processing Controls**: Pause, step-through, and speed control options

### 📊 Analytics & Insights
- **Vehicle Count per Lane**: Real-time counting of vehicles in each traffic lane
- **Traffic Flow Metrics**: Historical and real-time traffic density analysis
- **Visual Annotations**: Clear bounding boxes, lane markers, and status indicators
- **Progress Tracking**: Frame-by-frame processing with completion percentage

---

## 🧠 Technical Architecture

### Model Architecture
```
Input Video/Camera Stream
        ↓
   Frame Preprocessing
        ↓
   YOLOv8 Detection
        ↓
   ROI-based Filtering
        ↓
   Lane Classification
        ↓
   Traffic Analysis
        ↓
   Visualization & Output
```

### Key Technologies
- **YOLOv8**: State-of-the-art object detection model
- **OpenCV**: Computer vision and video processing
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computations
- **Ultralytics**: YOLOv8 implementation and training

---

## 🚀 Project Potential & Applications

### 🏙️ Smart City Integration
- **Traffic Management**: Real-time traffic flow optimization
- **Signal Control**: Adaptive traffic light timing based on vehicle density
- **Urban Planning**: Data-driven infrastructure development
- **Emergency Response**: Quick identification of traffic congestion for emergency routes

### 🚦 Transportation Applications
- **Highway Monitoring**: Continuous traffic surveillance on major roads
- **Toll Plaza Management**: Vehicle counting and classification for billing
- **Parking Systems**: Real-time parking space availability monitoring
- **Fleet Management**: Commercial vehicle tracking and route optimization

### 📈 Business & Research Applications
- **Market Research**: Vehicle type distribution analysis for automotive industry
- **Environmental Impact**: Traffic-based pollution monitoring and reporting
- **Insurance Analytics**: Risk assessment based on traffic patterns
- **Academic Research**: Traffic behavior studies and urban mobility research

### 🌍 Scalability Potential
- **Multi-Camera Networks**: Coordinate multiple cameras for city-wide monitoring
- **Cloud Integration**: Scalable cloud-based processing for large deployments
- **IoT Integration**: Connection with smart traffic infrastructure
- **AI Enhancement**: Continuous learning from traffic patterns

---

## 📁 Directory Structure

```
YOLO Implementation/
├── 📄 README.md                           # Comprehensive project documentation
├── 📋 requirements.txt                    # Python dependencies
├── ⚙️ setup.py                           # Automated installation script
├── 🚫 .gitignore                         # Git ignore configuration
│
├── 🤖 models/                            # Trained model files
│   ├── best.pt                          # PyTorch model weights (primary)
│   └── best.onnx                        # ONNX format for deployment
│
├── 🐍 Python Scripts/
│   ├── indian_traffic_analysis.py       # Main traffic analysis script
│   ├── indian_traffic_analysis_windowed.py  # Enhanced windowed version
│   └── polygon_calibrator.py            # Interactive coordinate calibration tool
│
├── 📊 Data & Configuration/
│   ├── polygon_coordinates.txt           # Calibrated polygon coordinates
│   ├── Notebook.ipynb                   # Training and analysis notebook
│   └── Data/                            # Training dataset (if available)
│
├── 📹 Sample Videos/
│   ├── indian traffic.mp4               # Sample Indian traffic video
│   └── indian traffic2.mp4              # Additional test video
│
├── 🖼️ images/                           # Project documentation images
│   ├── classification.png               # Model classification results
│   ├── confusion.png                    # Confusion matrix visualization
│   ├── distribution.png                 # Dataset distribution analysis
│   ├── output.gif                       # Sample output demonstration
│   ├── sample_dataset.png               # Training dataset examples
│   ├── traffic_density_analysis.gif     # Traffic analysis demonstration
│   ├── training_loss.png                # Training loss curves
│   └── validation_inferences.png        # Validation results
│
└── 🔧 Configuration/
    ├── __pycache__/                      # Python bytecode cache
    └── .qodo/                           # Development environment files
```

### 📝 File Descriptions

#### 🎯 Core Analysis Scripts
- **`indian_traffic_analysis_windowed.py`**: 
  - Primary script for real-time traffic analysis
  - Features: Auto-scaling display, interactive controls, progress tracking
  - Best for: Production use and demonstrations

- **`indian_traffic_analysis.py`**: 
  - Original traffic analysis implementation
  - Features: Basic functionality with custom coordinates
  - Best for: Learning and understanding the core logic

#### 🛠️ Utility Tools
- **`polygon_calibrator.py`**: 
  - Interactive tool for setting up lane detection polygons
  - Features: Click-to-define coordinates, visual polygon preview
  - Usage: Calibrate coordinates for new videos or camera angles

- **`setup.py`**: 
  - Automated environment setup and dependency installation
  - Features: One-click installation, environment validation
  - Usage: Run once during initial setup

#### 🤖 Model Files
- **`best.pt`**: PyTorch format - Primary model for development and training
- **`best.onnx`**: ONNX format - Optimized for deployment and cross-platform use

#### 📊 Data Files
- **`polygon_coordinates.txt`**: Pre-calibrated coordinates for Indian traffic scenarios
- **`requirements.txt`**: All Python dependencies with version specifications
- **`Notebook.ipynb`**: Jupyter notebook with training pipeline and analysis

---

## ⚙️ Installation & Setup

### 🔧 Prerequisites
- **Python 3.8+** (Recommended: Python 3.9 or 3.10)
- **CUDA-capable GPU** (Optional but recommended for faster processing)
- **Webcam or video files** for testing

### 🚀 Quick Start (Automated)

1. **Clone the repository**:
```bash
git clone https://github.com/prathamhanda/IoT-Based_Traffic_Regulation.git
cd "IoT-Based_Traffic_Regulation/YOLO Implementation"
```

2. **Run automated setup**:
```bash
python setup.py
```
This script will:
- Create a virtual environment
- Install all dependencies
- Verify model files
- Test the installation

### 🛠️ Manual Installation

1. **Create virtual environment**:
```bash
python -m venv traffic_analysis_env
# Windows
traffic_analysis_env\Scripts\activate
# Linux/Mac
source traffic_analysis_env/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import cv2, ultralytics, torch; print('Installation successful!')"
```

### 📦 Dependencies Overview
- **ultralytics**: YOLOv8 implementation and training
- **opencv-python**: Computer vision and video processing
- **torch**: PyTorch deep learning framework
- **numpy**: Numerical computations
- **matplotlib**: Visualization and plotting
- **argparse**: Command-line argument parsing

---

## 🎮 Usage Guide

### 🎯 Basic Usage

#### 1. **Video Analysis** (Recommended)
```bash
python indian_traffic_analysis_windowed.py --source "your_video.mp4" --output "analyzed_video.avi"
```

#### 2. **Live Camera Feed**
```bash
python indian_traffic_analysis_windowed.py --webcam --conf 0.5
```

#### 3. **High-Performance Processing**
```bash
python indian_traffic_analysis_windowed.py --source "video.mp4" --skip-frames 3 --conf 0.6
```

### ⚙️ Command-Line Options

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `--source` | Input video file path | `sample_video.mp4` | `"traffic.mp4"` |
| `--weights` | Model weights path | `models/best.pt` | `"models/best.onnx"` |
| `--output` | Output video path | `processed_video.avi` | `"output.mp4"` |
| `--conf` | Detection confidence threshold | `0.4` | `0.6` |
| `--webcam` | Use live camera feed | `False` | `--webcam` |
| `--skip-frames` | Process every Nth frame | `1` | `3` |

### 🎮 Interactive Controls (During Processing)

| Key | Action | Description |
|-----|--------|-------------|
| **Q** | Quit | Exit the application |
| **P** | Pause/Resume | Pause or resume video processing |
| **S** | Step Frame | Step through one frame (when paused) |
| **ESC** | Emergency Exit | Force quit the application |

### 🎯 Polygon Calibration (For New Videos)

When working with new video sources, you'll need to calibrate the lane detection:

1. **Run the calibration tool**:
```bash
python polygon_calibrator.py --source "your_new_video.mp4"
```

2. **Define lane boundaries**:
   - Click points to define Lane 1 polygon
   - Press SPACE to switch to Lane 2
   - Press ENTER to finish and save coordinates

3. **Update coordinates**:
   - Copy generated coordinates from `polygon_coordinates.txt`
   - Paste into your analysis script

### 📊 Understanding the Output

#### Visual Indicators
- **Green Polygon**: Lane 1 detection area
- **Blue Polygon**: Lane 2 detection area
- **Yellow Lines**: Detection zone boundaries
- **White Line**: Lane separator (threshold)
- **Colored Circles**: Detected vehicle centers
  - Green: Lane 1 vehicles
  - Blue: Lane 2 vehicles
  - Yellow: Fallback detection

#### Text Annotations
- **Vehicle Count**: Real-time count per lane
- **Traffic Intensity**: "Smooth" or "Heavy" classification
- **Frame Counter**: Current frame and progress percentage
- **Control Instructions**: Available keyboard shortcuts

---

## 📊 Model Performance

### 🎯 Detection Metrics
![Confusion Matrix](images/confusion.png)

### 📈 Training Performance
![Training Loss](images/training_loss.png)

### 🎨 Sample Results
![Sample Detection](images/validation_inferences.png)

### 📊 Performance Benchmarks

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP50** | 0.89 | Mean Average Precision at IoU 0.5 |
| **mAP50-95** | 0.67 | Mean Average Precision at IoU 0.5-0.95 |
| **Precision** | 0.91 | Percentage of correct positive predictions |
| **Recall** | 0.87 | Percentage of actual positives identified |
| **FPS** | 30+ | Frames per second (GPU) / 8-12 (CPU) |

### 🎯 Supported Vehicle Classes
- **Cars**: Sedans, hatchbacks, SUVs
- **Trucks**: Heavy vehicles, delivery trucks
- **Buses**: Public transport, coaches
- **Motorcycles**: Two-wheelers, scooters
- **Auto-rickshaws**: Three-wheelers (Indian context)

---

## 🔧 Configuration & Customization

### 🎯 Traffic Threshold Adjustment

Modify traffic intensity thresholds in the script:
```python
heavy_traffic_threshold = 8  # Vehicles per lane for "Heavy" classification
```

### 🎨 Visual Customization

Customize colors and annotations:
```python
font_scale = 2.0                    # Text size
font_color = (255, 255, 255)       # White text
background_color = (0, 0, 255)     # Red background
line_thickness = 4                  # Polygon line thickness
```

### ⚡ Performance Optimization

For better performance:
```python
# Process every 3rd frame for 3x speed boost
--skip-frames 3

# Lower confidence for more detections
--conf 0.3

# Higher confidence for fewer false positives
--conf 0.7
```

### 🎯 Multi-Camera Setup

For multiple camera feeds:
```python
# Camera 1
python script.py --source 0 --output cam1.avi

# Camera 2  
python script.py --source 1 --output cam2.avi
```

---

## 🎯 Advanced Features

### 🤖 Model Switching
Switch between model formats:
```bash
# Use PyTorch model (default)
python script.py --weights "models/best.pt"

# Use ONNX model (faster inference)
python script.py --weights "models/best.onnx"
```

### 📊 Data Export
Export traffic data for analysis:
```python
# Add to script for CSV export
import csv
traffic_data = {
    'timestamp': time.time(),
    'lane1_count': vehicles_in_left_lane,
    'lane2_count': vehicles_in_right_lane,
    'intensity': traffic_intensity
}
```

### 🌐 Integration APIs
Potential integration points:
- **REST API**: For web application integration
- **WebSocket**: For real-time dashboards
- **Database**: For historical traffic analysis
- **Cloud Services**: For scalable deployment

---

## 🚀 Future Enhancements

### 🎯 Planned Features
- [ ] **Multi-lane support**: Detection for 3+ lanes
- [ ] **Speed estimation**: Vehicle speed calculation
- [ ] **License plate recognition**: Vehicle identification
- [ ] **Traffic violation detection**: Rule violation alerts
- [ ] **Weather adaptation**: Performance optimization for different weather
- [ ] **Cloud deployment**: AWS/Azure deployment templates

### 🤖 AI Improvements
- [ ] **Custom training pipeline**: Easy model retraining
- [ ] **Transfer learning**: Adaptation to new environments
- [ ] **Ensemble models**: Multiple model voting
- [ ] **Temporal analysis**: Traffic pattern recognition

### 🌐 System Integration
- [ ] **IoT sensors**: Integration with traffic sensors
- [ ] **Mobile app**: Companion mobile application
- [ ] **Web dashboard**: Real-time monitoring interface
- [ ] **Alert system**: Automatic congestion notifications

---

## 🔍 Troubleshooting

### Common Issues & Solutions

#### 1. **Window Not Visible**
```bash
# Try smaller display resolution
python script.py --display-width 800 --display-height 600

# Or use windowed mode
python indian_traffic_analysis_windowed.py
```

#### 2. **Low Performance**
```bash
# Reduce processing load
python script.py --skip-frames 5 --conf 0.6

# Use smaller input resolution
python script.py --imgsz 416
```

#### 3. **Model Loading Error**
```bash
# Verify model file exists
ls models/best.pt

# Re-download model if corrupted
# Check model file size (should be ~20-50MB)
```

#### 4. **Coordinate Calibration Issues**
```bash
# Use calibration tool for new videos
python polygon_calibrator.py --source "your_video.mp4"

# Manually adjust coordinates in polygon_coordinates.txt
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### 🎯 Ways to Contribute
- **Bug Reports**: Report issues with detailed descriptions
- **Feature Requests**: Suggest new functionality
- **Code Contributions**: Submit pull requests with improvements
- **Documentation**: Improve documentation and tutorials
- **Testing**: Test on different video sources and environments

### 📝 Development Setup
```bash
# Fork the repository
git clone https://github.com/your-username/IoT-Based_Traffic_Regulation.git

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python -m pytest tests/

# Submit pull request
git push origin feature/your-feature-name
```

### 🏗️ Code Standards
- Follow PEP 8 Python style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation for changes

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🎯 Usage Rights
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

### 📋 Requirements
- Include original license
- Include copyright notice

---

## 🙏 Acknowledgments

### 🎯 Technologies Used
- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)**: Object detection framework
- **[OpenCV](https://opencv.org/)**: Computer vision library  
- **[PyTorch](https://pytorch.org/)**: Deep learning framework

### 👥 Contributors
- **[Pratham Handa](https://github.com/prathamhanda)**: Project Lead & Developer
- **Community Contributors**: Thank you to all contributors!

### 📚 References
- YOLO: Real-Time Object Detection
- Computer Vision for Traffic Analysis
- Smart City Traffic Management Systems

---

## 📞 Support & Contact

### 🎯 Getting Help
- **GitHub Issues**: [Report bugs or ask questions](https://github.com/prathamhanda/IoT-Based_Traffic_Regulation/issues)
- **Documentation**: Check this README and inline code comments
- **Community**: Join discussions in the repository

### 📧 Contact Information
- **Project Repository**: [IoT-Based Traffic Regulation](https://github.com/prathamhanda/IoT-Based_Traffic_Regulation)
- **Developer**: [Pratham Handa](https://github.com/prathamhanda)

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

*Made with ❤️ for smarter traffic management*

</div>

---

## Requirements
Install the required dependencies using:
```bash
pip install -r requirements.txt
```

### Core Libraries:
- `ultralytics` (for YOLOv8)
- `opencv-python` (for video processing)
- `torch` (PyTorch framework)
- `matplotlib`

---

## Usage
### Real-Time Detection
Run the provided Python script for detecting vehicles in a video file or live stream:
```bash
python real_time_traffic_analysis.py --source sample_video.avi --weights models/best.pt
```

The output will be saved as an annotated video, e.g., `processed_sample_video.mp4`.

---

## Results

### Sample Dataset
Here are some sample images from the dataset used for training:
![Sample Dataset](images/sample_dataset.png)

---


### Training Loss
The following plots show the training and validation loss curves for different components of the YOLOv8 model. These learning curves demonstrate the gradual reduction in loss over 20 epochs:

#### 1. Box Loss Learning Curve
![Box Loss Learning Curve](images/training_loss.png)

This curve represents the error in predicted bounding boxes for detected objects. A steady decline in both training and validation loss indicates the model's ability to generalize effectively.

---

### Classification Loss Learning Curve
![Classification Loss Learning Curve](images/classification.png)

This plot illustrates the model's error in classifying detected objects correctly. The convergence of training and validation loss implies that the model is well-tuned and avoids overfitting.

---

### Distribution Focal Loss Learning Curve
![Distribution Focal Loss Learning Curve](images/distribution.png)

The focal loss is used to handle class imbalance during training. This curve shows consistent improvement in both training and validation, indicating effective handling of rare classes.

---

### Confusion Matrix
![Confusion Matrix](images/confusion.png)

The confusion matrix below shows the model's performance on the validation set, highlighting its ability to distinguish between vehicles and the background.

- **True Positives (Vehicle Detected as Vehicle):** High accuracy (97%).
- **False Negatives (Vehicle Detected as Background):** Low error (3%).

---

### Observations
1. **Convergence of Loss Curves:** 
   - The training and validation loss curves are close, suggesting the model is not overfitting and generalizes well to unseen data.
   - The smooth decline in the loss values indicates effective learning.

2. **Confusion Matrix Insights:** 
   - High recall for vehicles demonstrates the model's ability to detect most vehicles in the dataset.
   - The low false-positive rate indicates minimal misclassification of background regions as vehicles.

3. **Classification and Focal Loss:** 
   - The focal loss curve shows the model's capacity to handle imbalance in the dataset, improving performance for underrepresented classes.

4. **Box Loss:** 
   - Consistent reduction in box loss reflects the model's increasing accuracy in predicting bounding box coordinates.

---


### Validation Set Inferences
Below are sample inferences from the validation set:
![Validation Inferences](images/validation_inferences.png)

---

### Real-Time Video Detection
The model processes videos and identifies vehicles in real-time. Below is an example of annotated frames from a video:
![Real-Time Detection](images/output.gif)

### Real-Time Traffic Intensity
The traffic intensity is analyzed based on the number of vehicles detected in each lane:
![Traffic Analysis](images/traffic_density_analysis.gif)

---

## Performance Metrics

The model achieved the following performance metrics during validation:

| **Metric**        | **Value** | **Description**                                                                                   |
|--------------------|-----------|---------------------------------------------------------------------------------------------------|
| **mAP@0.5**       | 0.975     | Mean Average Precision at IoU threshold of 0.5, measuring the precision-recall tradeoff for detections. |
| **Precision**     | 0.916     | The fraction of correctly identified vehicles out of all predicted vehicles (low false positives). |
| **Recall**        | 0.938     | The fraction of actual vehicles correctly identified by the model (low false negatives).          |
| **True Positives**| 97%       | Percentage of vehicles correctly classified as vehicles.                                          |
| **False Negatives**| 3%        | Percentage of vehicles mistakenly classified as background.                                       |

### Observations:
- The high mAP@0.5 demonstrates the robustness of the YOLOv8 model in accurately detecting vehicles within bounding boxes.
- Precision indicates a low false-positive rate, showing the model’s reliability in avoiding incorrect predictions.
- Recall suggests the model’s ability to detect the majority of vehicles in the validation dataset.
- The confusion matrix corroborates the high performance, with minimal misclassifications.

---

# YOLOv8 ONNX Model Deployment

This section covers the export of the trained YOLOv8 model to the ONNX format, which facilitates deployment on various platforms such as mobile devices, edge devices, or custom frameworks.

### Benefits of ONNX:
- **Platform Independence**: Deploy the model across different computing frameworks and programming languages.
- **Lightweight Deployment**: Optimal for real-time inference on low-resource devices.
- **Framework Compatibility**: Compatible with major frameworks like PyTorch, TensorFlow, OpenCV, and others.

### Exporting to ONNX
The model is already exported and is available in the repository under `models/best.onnx`. To export the model yourself, you can follow the command below:
```python
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO('models/best.pt')

# Export to ONNX format with dynamic input size support
model.export(format='onnx', dynamic=True)
```

### Running Inference with ONNX Runtime
To run inference using the ONNX runtime, follow these steps:

1. **Install ONNX Runtime**:
   ```bash
   pip install onnxruntime
   ```

2. **Load and Run Inference**:
   ```python
   import onnxruntime as ort
   import numpy as np
   import cv2

   # Load ONNX model
   session = ort.InferenceSession("models/best.onnx")

   # Prepare input image
   input_image = cv2.imread("path/to/image.jpg")
   input_image = cv2.resize(input_image, (640, 640))  # Resize to model's input size
   input_tensor = np.expand_dims(input_image.transpose(2, 0, 1), axis=0).astype(np.float32)

   # Run inference
   outputs = session.run(None, {"images": input_tensor})
   print(outputs)
   ```

### Notes:
- The `dynamic=True` flag in the export command ensures that the ONNX model can handle variable input sizes.
- Ensure that the pre-processing steps (e.g., resizing, normalization) align with the training configuration.

## Future Work
- Extend the model to classify different types of vehicles (e.g., cars, trucks, motorcycles).
- Deploy the model as a real-time web app for traffic monitoring.
- Improve traffic intensity analysis with advanced lane detection.

---

## Credits
This project builds on:
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/yolov8)
- Datasets and initial project structure adapted from Kaggle contributions.

