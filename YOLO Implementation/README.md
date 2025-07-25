# 🚗 Real-Time Vehicle Detection and Traffic Flow Classification System

Complete traffic analysis system with AI-powered vehicle detection, lane monitoring, and multi-source input support including YouTube live streams.

## ✨ Features

- **🎯 Advanced AI Detection**: YOLOv8-powered vehicle detection with high accuracy
- **�️ Violence Detection**: Real-time violence, gore, and weapon detection using Sightengine API
- **�📺 Multi-Source Input**: Support for webcams, video files, IP cameras, RTSP streams, and **YouTube live streams**
- **🛣️ Interactive Lane Calibration**: Point-and-click polygon creation for custom lane detection
- **📊 Real-Time Analytics**: Live traffic flow analysis, density monitoring, and statistics
- **🔄 Auto-Reconnection**: Intelligent stream handling with automatic reconnection for live sources
- **⚙️ Flexible Configuration**: Extensive customization options and preset configurations
- **🖥️ Dual Mode Operation**: GUI mode for interactive use, headless mode for server deployment
- **🛠️ System Tools**: Comprehensive diagnostics, testing utilities, and performance benchmarks
- **📝 Evidence System**: Automatic saving of violence detection incidents with metadata

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ 
- OpenCV 4.8+ with GUI support
- CUDA-capable GPU (optional, for acceleration)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/prathamhanda/IoT-Based_Traffic_Regulation.git
cd "YOLO Implementation"
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download YOLO model:**
   - Place your trained YOLOv8 model as `models/best.pt`
   - Or use the provided pre-trained model

4. **Setup Violence Detection (Optional):**
```bash
python direct_setup.py
```
   - Get API credentials from [Sightengine](https://sightengine.com/)
   - Follow the interactive setup process

### Basic Usage

**System Requirements Check:**
```bash
python config.py --check-system
```

**Setup Violence Detection:**
```bash
python direct_setup.py --direct
```

**Start Traffic Analysis:**
```bash
python traffic_analysis.py
```

**YouTube Live Stream:**
```bash
python traffic_analysis.py --source "https://www.youtube.com/live/YOUR_STREAM_URL"
```

**Headless Mode (Server):**
```bash
python traffic_analysis.py --headless --source YOUR_SOURCE --output analysis_output.avi
```

## 📋 Main Components

### 1. `traffic_analysis.py` - Main Analysis Engine
Complete traffic analysis system with all features integrated:

**Core Features:**
- YouTube stream support with auto-reconnection
- Interactive polygon calibration
- Real-time vehicle detection and counting
- Lane-wise traffic analysis
- Multi-backend video capture

**Example Commands:**
```bash
# Interactive mode with webcam
python traffic_analysis.py --source 0

# YouTube live stream analysis
python traffic_analysis.py --source "https://www.youtube.com/live/YOUR_STREAM_URL"

# Video file with custom confidence
python traffic_analysis.py --source "video.mp4" --conf 0.5

# Headless server mode
python traffic_analysis.py --headless --source rtsp://camera_ip --output output.avi
```

### 2. `system_tools.py` - Diagnostics & Testing
Comprehensive system checking and debugging utilities:

**Usage Examples:**
```bash
# Complete system check
python system_tools.py --check

# Test all video sources
python system_tools.py --test-sources

# Test specific stream
python system_tools.py --test-stream "https://youtube.com/live/..."

# Performance benchmark
python system_tools.py --benchmark
```

### 3. `config.py` - Configuration Management
Centralized configuration system with presets:

**Usage Examples:**
```bash
# Show current configuration
python config.py --show

# Load performance preset
python config.py --preset performance

# Custom settings
python config.py --confidence 0.6 --threshold 10 --save
```

## 🛡️ Violence Detection

The system includes advanced content moderation using the Sightengine API to detect violence, gore, and weapons in real-time.

### Setup Violence Detection

1. **Get API Credentials:**
   - Visit [Sightengine](https://sightengine.com/) and create a free account
   - Get your API User ID and Secret from the dashboard

2. **Configure Violence Detection:**
```bash
python direct_setup.py
```

3. **Test API Connection:**
```bash
python direct_setup.py --check
```

### Violence Detection Features

- **Real-time Analysis**: Asynchronous processing without performance impact
- **Multiple Models**: Violence, gore, and weapon detection
- **Configurable Thresholds**: Adjustable sensitivity (0.1-1.0)
- **Evidence Preservation**: Automatic saving of detected incidents
- **Alert System**: Visual alerts and comprehensive logging
- **Statistics**: Detection rates and performance metrics

### Configuration Options

```bash
# Set detection threshold
python traffic_analysis.py --violence-threshold 0.8

# Set check interval (frames)
python traffic_analysis.py --violence-interval 60

# Disable violence detection
python traffic_analysis.py --disable-violence
```

### Evidence Storage

Detected incidents are automatically saved to:
- `evidence/violence/` - Violence incidents
- `evidence/gore/` - Gore detection
- `evidence/weapons/` - Weapon detection
- `logs/violence_detection.log` - Detection logs
- `logs/violence_alerts.log` - Alert logs

## 🛣️ Lane Calibration Guide

Interactive calibration tool for setting up custom lane detection:

1. **Start Calibration:** Run `python traffic_analysis.py --source YOUR_SOURCE`
2. **Create Lane Polygons:** Left-click to add points, right-click to complete
3. **Controls:**
   - `Left Click`: Add calibration point
   - `Right Click`: Complete current polygon
   - `R`: Reset current polygon
   - `C`: Complete calibration
   - `Q`: Cancel calibration

## 📺 YouTube Live Stream Support

Advanced YouTube integration with automatic URL extraction and reconnection:

```bash
# YouTube live stream
python traffic_analysis.py --source "https://www.youtube.com/live/STREAM_ID"

# Test YouTube connection
python system_tools.py --test-stream "YOUR_YOUTUBE_URL"
```

## 🚨 Troubleshooting

### Common Issues:

**1. Model Not Found:**
- Ensure YOLO model is in `models/best.pt`

**2. Video Source Issues:**
- Test with: `python system_tools.py --test-sources`

**3. YouTube Connection Failed:**
- Update yt-dlp: `pip install --upgrade yt-dlp`
- Test connection: `python system_tools.py --test-stream "URL"`

**4. GUI Issues:**
- Install OpenCV with GUI support or use `--headless`

### System Requirements Check:
```bash
python system_tools.py --check
```

## 🔧 Configuration Options

### Available Presets:
- `default`: Balanced settings for general use
- `high-accuracy`: Higher confidence thresholds
- `performance`: Optimized for speed
- `server`: Headless server deployment
- `demo`: Lightweight for demonstrations

## 📊 Output and Analytics

### Real-Time Display:
- Vehicle detection boxes with confidence scores
- Lane polygons with color coding
- Traffic count per lane
- Traffic intensity status (Smooth/Heavy)
- Frame rate (FPS) indicator

### Statistics:
- Total frames processed
- Total vehicles detected
- Average FPS
- Analysis duration

## 👨‍💻 Author

**Pratham Handa**
- GitHub: [@prathamhanda](https://github.com/prathamhanda)
- Project: [IoT-Based Traffic Regulation](https://github.com/prathamhanda/IoT-Based_Traffic_Regulation)

---

**Happy Traffic Monitoring! 🚗📊**
