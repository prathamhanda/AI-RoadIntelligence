# Video Files Setup Guide

## 📹 Required Video Files

Due to GitHub's file size limitations, video files are not included in this repository. You'll need to add your own video files for testing.

### 🎯 Expected Video Files

Place the following files in the main directory:

```
YOLO Implementation/
├── indian traffic.mp4          # Primary test video (Indian traffic scenario)
├── indian traffic2.mp4         # Secondary test video (optional)
└── your_custom_video.mp4       # Any custom video for testing
```

### 📋 Video Requirements

- **Format**: MP4, AVI, MOV, or other OpenCV-supported formats
- **Resolution**: Any resolution (auto-scaling implemented)
- **Content**: Traffic scenes with vehicles
- **Duration**: Any duration (processing shows progress)

### 🎮 How to Use with Videos

1. **Add your video file** to the YOLO Implementation directory
2. **Run the analysis**:
   ```bash
   python indian_traffic_analysis_windowed.py --source "your_video.mp4"
   ```

3. **For optimal performance**:
   ```bash
   # Skip frames for faster processing
   python indian_traffic_analysis_windowed.py --source "your_video.mp4" --skip-frames 3
   ```

### 🔧 Custom Coordinate Calibration

If using new videos, calibrate the lane detection:

1. **Run calibration tool**:
   ```bash
   python polygon_calibrator.py --source "your_video.mp4"
   ```

2. **Click to define lane boundaries**
3. **Copy generated coordinates** from `polygon_coordinates.txt`
4. **Update coordinates** in the analysis script

### 📊 Sample Videos for Testing

You can use any traffic videos for testing. Good sources include:
- Personal traffic recordings
- Dashcam footage
- Public traffic camera feeds
- YouTube traffic videos (downloaded legally)

### ⚠️ Important Notes

- Large video files (>100MB) should not be committed to git
- The `.gitignore` file automatically excludes `.mp4` files
- Use the `--skip-frames` parameter for large videos to improve performance
- Videos are processed locally and not uploaded to GitHub

### 🎯 Alternative Testing

If you don't have video files, you can:
- Use webcam feed: `python script.py --webcam`
- Download sample traffic videos from public sources
- Use the included sample images for basic testing
