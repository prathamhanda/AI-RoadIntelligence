# 🚦 IoT-Based Traffic Regulation System

**Complete Integrated Smart Traffic Management System with Real-time Detection and Automated Control**

## 🎯 Project Overview

This project implements a comprehensive IoT-based traffic regulation system that combines:

1. **🚗 Real-time Vehicle Detection** - Using YOLOv8 for accurate vehicle counting and lane analysis
2. **🐕 Animal Detection & Safety** - Automatic detection of animals on roads with traffic adjustments
3. **🛡️ Violence Detection** - Advanced AI-powered violence detection with emergency alerts
4. **🚦 Automated Traffic Control** - SUMO-based traffic light control and flow optimization
5. **📲 Emergency Response** - SMS alerts to authorities for critical situations

## ✨ Key Features

### 🎥 Multi-Source Input Support
- YouTube live streams with auto-reconnection
- Webcam support with automatic camera detection
- Video files (MP4, AVI, etc.)
- IP cameras and RTSP streams

### 🎯 Interactive Calibration
- User-friendly polygon calibration for lane detection
- Real-time preview during calibration
- Automatic configuration saving/loading

### 🤖 AI-Powered Detection Systems
- **Vehicle Detection**: YOLOv8-based vehicle counting per lane
- **Animal Detection**: Specialized model for animal safety
- **Violence Detection**: Cloud-based AI for public safety

### 🚦 Smart Traffic Control
- **SUMO Integration**: Professional traffic simulation
- **Adaptive Traffic Lights**: Dynamic control based on vehicle density
- **Emergency Protocols**: Automatic adjustments for safety incidents
- **Real-time Monitoring**: Live status display and control

### 📱 Emergency Response System
- **SMS Alerts**: Twilio integration for emergency notifications
- **Configurable Thresholds**: Customizable alert conditions
- **Multi-level Responses**: Different actions for different severity levels

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/prathamhanda/IoT-Based_Traffic_Regulation.git
cd IoT-Based_Traffic_Regulation/YOLO\ Implementation

# Install Python dependencies
pip install -r requirements_integrated.txt

# Optional: Install SUMO for traffic simulation
# Download from: https://sumo.dlr.de/docs/Installing/index.html
```

### 2. Configuration

```bash
# Set up API credentials for violence detection (optional)
python direct_setup.py

# Configure Twilio for SMS alerts (optional)
# Edit the TWILIO_* variables in traffic_analysis.py
```

### 3. Run the System

```bash
# Run with interactive source selection
python traffic_analysis.py

# Run with specific video file
python traffic_analysis.py --source videos/traffic_video.mp4

# Run with webcam
python traffic_analysis.py --source 0

# Run with YouTube live stream
python traffic_analysis.py --source "https://youtube.com/live/stream_id"

# Run in headless mode (no GUI)
python traffic_analysis.py --headless --source video.mp4
```

## 🎮 Usage Guide

### Step 1: Source Selection
Choose your input source:
1. Video file
2. Webcam (automatic detection)
3. IP Camera/HTTP Stream
4. RTSP Stream
5. YouTube Live Stream

### Step 2: Lane Calibration
- Left-click to add polygon points
- Right-click to complete polygon (minimum 3 points)
- Create at least 2 lane polygons
- Press 'c' to complete calibration

### Step 3: Real-time Analysis
The system will display:
- **Vehicle counts** per lane
- **Traffic density** status
- **Violence detection** alerts
- **Animal detection** warnings
- **Traffic control** status
- **Emergency response** indicators

## 📊 Real-time Display Features

### Vehicle Detection Overlay
- Green/Red polygons for each lane
- Real-time vehicle count display
- Traffic intensity indicators (Heavy/Smooth)

### Safety Monitoring Dashboard
- **Violence Monitor**: Real-time threat detection status
- **Animal Monitor**: Animal presence and safety alerts
- **Traffic Control**: Automated system status and actions

### Emergency Alert System
- **Visual Alerts**: On-screen emergency notifications
- **SMS Notifications**: Automatic emergency response
- **Traffic Adjustments**: Immediate safety measures

## ⚙️ Command Line Options

### Basic Options
```bash
--source VIDEO_SOURCE          # Video source (file, webcam, URL)
--weights MODEL_PATH           # YOLO model weights path
--conf THRESHOLD              # Detection confidence threshold
--headless                    # Run without GUI
--no-calibration             # Skip calibration, use saved config
--output OUTPUT_FILE          # Save processed video
```

### Traffic Control Options
```bash
--disable-traffic-control      # Disable automated traffic control
--vehicle-threshold NUMBER     # Vehicle count for traffic control
--violence-alert-duration SEC  # Violence detection alert threshold
--animal-alert-duration SEC    # Animal detection alert threshold
--emergency-number PHONE       # Emergency SMS number
```

### Violence Detection Options
```bash
--violence-api-user USER       # Sightengine API user ID
--violence-api-secret SECRET   # Sightengine API secret
--disable-violence            # Disable violence detection
--violence-threshold FLOAT    # Violence detection threshold
```

## 🛠️ System Architecture

```
Input Sources → Calibration → Detection Systems → Traffic Control → Emergency Response
     ↓              ↓              ↓                ↓                 ↓
• Video Files   • Interactive   • Vehicle Counting  • SUMO Control   • SMS Alerts
• Webcam        • Polygon       • Animal Detection  • Light Control  • Visual Alerts
• IP Cameras    • Setup         • Violence Monitor  • Flow Optimize  • Auto Response
• YouTube       • Auto-save     • Real-time AI      • Emergency Stop • Multi-level
```

## 📈 Performance Features

- **Real-time Processing**: Optimized for live video analysis
- **Multi-threading**: Parallel processing for better performance
- **GPU Acceleration**: CUDA support for faster inference
- **Adaptive FPS**: Dynamic frame rate adjustment
- **Memory Management**: Efficient resource utilization

## 🔧 Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Check camera connections
   - Close other applications using camera
   - Try different camera IDs (0, 1, 2...)

2. **SUMO Not Working**
   - Install SUMO from official website
   - Ensure `sumo-gui` is in system PATH
   - Check simulation files in `simulation_files/`

3. **Violence Detection Disabled**
   - Set up Sightengine API credentials
   - Run `python direct_setup.py` for configuration

4. **SMS Alerts Not Working**
   - Configure Twilio credentials
   - Verify phone number format (+country code)

## 📚 Dependencies

### Core Requirements
- Python 3.8+
- OpenCV 4.8+
- Ultralytics YOLOv8
- NumPy

### Optional Components
- **SUMO**: Traffic simulation (download separately)
- **Twilio**: SMS alert functionality
- **yt-dlp**: YouTube stream support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Pratham Handa** - *Main Developer* - [GitHub](https://github.com/prathamhanda)

## 🙏 Acknowledgments

- Ultralytics for YOLOv8
- SUMO Traffic Simulation
- Sightengine for violence detection API
- Twilio for SMS services
- OpenCV community

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact: [prathamhanda@example.com]

---

**🚦 Making Roads Safer with AI-Powered Traffic Intelligence**
