#!/usr/bin/env python3
"""
🚗 Real-Time Vehicle Detection and Traffic Flow Classification System
Complete traffic analysis system with YouTube live streams, webcam, video files, and IP cameras

Features:
- YouTube live stream support with auto-reconnection
- Multi-source input (webcam, video files, IP cameras, RTSP)
- Interactive lane calibration
- Real-time vehicle detection using YOLOv8
- Traffic flow analysis and density monitoring
- Headless mode for server deployment
- Automatic error handling and reconnection

Author: Pratham Handa
GitHub: https://github.com/prathamhanda/IoT-Based_Traffic_Regulation
"""

import cv2
import numpy as np
import argparse
import os
import time
import threading
from queue import Queue
from ultralytics import YOLO
import json
from datetime import datetime

class YouTubeStreamManager:
    """Enhanced YouTube stream handling with automatic reconnection"""
    
    def __init__(self, youtube_url, refresh_interval=300):
        self.youtube_url = youtube_url
        self.refresh_interval = refresh_interval
        self.current_stream_url = None
        self.last_refresh = 0
        self.cap = None
        self.consecutive_failures = 0
        self.max_failures = 5
        
    def get_fresh_stream_url(self):
        """Get fresh stream URL from YouTube using yt-dlp"""
        try:
            import yt_dlp
            
            print("🔄 Refreshing YouTube stream URL...")
            ydl_opts = {
                'format': 'best[height<=720][fps<=30]',
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
                stream_url = info.get('url')
                
                if stream_url:
                    print("✅ Fresh YouTube stream URL obtained")
                    print(f"📺 Title: {info.get('title', 'Unknown')}")
                    if info.get('is_live'):
                        print(f"🔴 Live stream - View count: {info.get('view_count', 'Unknown')}")
                    self.last_refresh = time.time()
                    self.consecutive_failures = 0
                    return stream_url
                else:
                    print("❌ Could not extract stream URL from YouTube")
                    return None
                    
        except ImportError:
            print("❌ yt-dlp not installed. Install with: pip install yt-dlp")
            return None
        except Exception as e:
            print(f"❌ Error extracting YouTube stream: {e}")
            print("   Try updating yt-dlp: pip install --upgrade yt-dlp")
            return None
    
    def should_refresh_url(self):
        """Check if URL should be refreshed"""
        time_since_refresh = time.time() - self.last_refresh
        return (time_since_refresh > self.refresh_interval or 
                self.consecutive_failures > 2)
    
    def connect(self):
        """Connect with fresh URL if needed"""
        if self.current_stream_url is None or self.should_refresh_url():
            self.current_stream_url = self.get_fresh_stream_url()
            
            if not self.current_stream_url:
                return False
        
        try:
            if self.cap:
                self.cap.release()
                
            self.cap = cv2.VideoCapture(self.current_stream_url, cv2.CAP_FFMPEG)
            
            if self.cap.isOpened():
                # Optimize settings for live streams
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                
                # Test read
                ret, frame = self.cap.read()
                if ret:
                    print("✅ YouTube stream connected successfully")
                    self.consecutive_failures = 0
                    return True
                    
            print("❌ Failed to connect to YouTube stream")
            self.consecutive_failures += 1
            return False
            
        except Exception as e:
            print(f"❌ YouTube connection error: {e}")
            self.consecutive_failures += 1
            return False
    
    def read(self):
        """Read frame with auto-reconnection"""
        if not self.cap or not self.cap.isOpened():
            if not self.connect():
                return False, None
        
        ret, frame = self.cap.read()
        
        if not ret:
            self.consecutive_failures += 1
            if self.consecutive_failures < self.max_failures:
                print(f"⚠️  Read failed (attempt {self.consecutive_failures}), reconnecting...")
                time.sleep(1)
                if self.connect():
                    ret, frame = self.cap.read()
            else:
                print("❌ Max YouTube failures reached")
        else:
            self.consecutive_failures = 0
            
        return ret, frame
    
    def release(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
            self.cap = None

class StreamCalibrator:
    """Interactive polygon calibration for traffic lanes"""
    
    def __init__(self, cap):
        self.cap = cap
        self.current_polygon = []
        self.polygons = []
        self.frame = None
        self.original_frame = None
        self.display_scale = 1.0
        self.display_width = 0
        self.display_height = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coordinates back to original frame coordinates
            if self.display_scale != 1.0:
                orig_x = int(x / self.display_scale)
                orig_y = int(y / self.display_scale)
            else:
                orig_x, orig_y = x, y
                
            self.current_polygon.append((orig_x, orig_y))
            print(f"Point added: ({orig_x}, {orig_y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_polygon) >= 3:
                self.polygons.append(self.current_polygon.copy())
                print(f"Lane {len(self.polygons)} completed with {len(self.current_polygon)} points")
                self.current_polygon = []
            else:
                print("Need at least 3 points for a polygon")
    
    def draw_polygons(self):
        self.frame = self.original_frame.copy()
        
        # Draw completed polygons
        for i, poly in enumerate(self.polygons):
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32)
                color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
                cv2.polylines(self.frame, [pts], True, color, 3)
                
                center = np.mean(pts, axis=0).astype(int)
                cv2.putText(self.frame, f"Lane {i+1}", tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw current polygon
        if len(self.current_polygon) > 0:
            for point in self.current_polygon:
                cv2.circle(self.frame, point, 5, (0, 255, 255), -1)
            
            if len(self.current_polygon) > 1:
                pts = np.array(self.current_polygon, dtype=np.int32)
                cv2.polylines(self.frame, [pts], False, (0, 255, 255), 2)
    
    def calibrate_stream(self):
        """Interactive polygon calibration"""
        print("\n=== LANE CALIBRATION ===")
        print("Instructions:")
        print("- Left click to add points to create lane polygons")
        print("- Right click to complete current polygon (min 3 points)")
        print("- Create at least 2 lane polygons")
        print("- Press 'c' to complete calibration")
        print("- Press 'r' to reset current polygon")
        print("- Press 'q' to quit")
        
        # Get frame dimensions for proper scaling
        ret, sample_frame = self.cap.read()
        if not ret:
            print("Failed to read sample frame!")
            return None
        
        frame_height, frame_width = sample_frame.shape[:2]
        
        # Calculate display scale for better compatibility
        max_display_width = 1024
        max_display_height = 576
        
        if frame_width > max_display_width or frame_height > max_display_height:
            width_scale = max_display_width / frame_width
            height_scale = max_display_height / frame_height
            self.display_scale = min(width_scale, height_scale)
            self.display_width = int(frame_width * self.display_scale)
            self.display_height = int(frame_height * self.display_scale)
        else:
            self.display_scale = 1.0
            self.display_width = frame_width
            self.display_height = frame_height
        
        # Setup calibration window
        window_name = 'Lane Calibration'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        cv2.moveWindow(window_name, 100, 50)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from stream!")
                return None
                
            self.original_frame = frame.copy()
            self.draw_polygons()
            
            # Add instructions overlay
            cv2.putText(self.frame, f"Polygons: {len(self.polygons)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(self.frame, "Press 'c' to complete, 'r' to reset", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Scale frame for display if needed
            if self.display_scale != 1.0:
                display_frame = cv2.resize(self.frame, (self.display_width, self.display_height))
            else:
                display_frame = self.frame
            
            cv2.imshow(window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if len(self.polygons) >= 2:
                    break
                else:
                    print("Please create at least 2 polygons before completing!")
            elif key == ord('r'):
                self.current_polygon = []
                print("Current polygon reset")
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        # Calculate lane threshold and detection area
        poly1_center_x = np.mean([p[0] for p in self.polygons[0]])
        poly2_center_x = np.mean([p[0] for p in self.polygons[1]])
        lane_threshold = int((poly1_center_x + poly2_center_x) / 2)
        
        all_y_coords = []
        for poly in self.polygons:
            all_y_coords.extend([p[1] for p in poly])
        
        min_y = min(all_y_coords)
        max_y = max(all_y_coords)
        x1 = max(0, min_y - 50)
        x2 = min(frame.shape[0], max_y + 50)
        
        config = {
            'polygons': self.polygons,
            'lane_threshold': lane_threshold,
            'detection_area': (x1, x2)
        }
        
        # Save configuration
        self.save_config(config)
        return config
    
    def save_config(self, config):
        """Save calibration configuration to file"""
        try:
            config_data = {
                'timestamp': datetime.now().isoformat(),
                'polygons': config['polygons'],
                'lane_threshold': config['lane_threshold'],
                'detection_area': config['detection_area']
            }
            
            with open('lane_config.json', 'w') as f:
                json.dump(config_data, f, indent=2)
            print("✅ Lane configuration saved to lane_config.json")
        except Exception as e:
            print(f"⚠️  Could not save configuration: {e}")

class TrafficAnalyzer:
    """Main traffic analysis processor"""
    
    def __init__(self, model, config, headless=False):
        self.model = model
        self.config = config
        self.headless = headless
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.processing = False
        self.stats = {
            'total_frames': 0,
            'vehicles_detected': 0,
            'analysis_start': time.time()
        }
        
    def analyze_frame(self, frame):
        """Analyze single frame for vehicle detection"""
        self.stats['total_frames'] += 1
        
        detection_frame = frame.copy()
        x1, x2 = self.config['detection_area']
        
        # Mask non-detection areas
        detection_frame[:x1, :] = 0
        detection_frame[x2:, :] = 0
        
        # Run YOLO inference
        results = self.model.predict(detection_frame, imgsz=640, conf=0.4, verbose=False)
        
        if not self.headless:
            processed_frame = results[0].plot(line_width=1)
            # Restore original areas
            processed_frame[:x1, :] = frame[:x1, :].copy()
            processed_frame[x2:, :] = frame[x2:, :].copy()
        else:
            processed_frame = frame.copy()
        
        # Draw lane polygons
        for i, poly in enumerate(self.config['polygons']):
            pts = np.array(poly, dtype=np.int32)
            color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
            cv2.polylines(processed_frame, [pts], True, color, 2)
        
        # Count vehicles in each lane
        vehicles_left = 0
        vehicles_right = 0
        lane_threshold = self.config['lane_threshold']
        
        if results[0].boxes is not None:
            total_vehicles = len(results[0].boxes)
            self.stats['vehicles_detected'] += total_vehicles
            
            for box in results[0].boxes.xyxy:
                if box[0] < lane_threshold:
                    vehicles_left += 1
                else:
                    vehicles_right += 1
        
        # Add traffic analysis annotations
        self.add_annotations(processed_frame, vehicles_left, vehicles_right)
        
        return processed_frame, vehicles_left, vehicles_right
    
    def add_annotations(self, frame, vehicles_left, vehicles_right):
        """Add traffic analysis overlays"""
        heavy_threshold = 8
        
        # Traffic intensity
        intensity_left = "Heavy" if vehicles_left > heavy_threshold else "Smooth"
        intensity_right = "Heavy" if vehicles_right > heavy_threshold else "Smooth"
        
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate font size based on frame size
        font_scale = min(frame_width / 1920, frame_height / 1080) * 1.5
        font_scale = max(0.5, min(font_scale, 2.0))
        
        # Left lane info box
        box_width = int(frame_width * 0.25)
        box_height = 70
        
        cv2.rectangle(frame, (20, 20), (20 + box_width, 20 + box_height), (0, 0, 255), -1)
        cv2.putText(frame, f'Left Lane: {vehicles_left} vehicles', (30, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Status: {intensity_left}', (30, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), 2)
        
        # Right lane info box
        right_x = frame_width - box_width - 20
        cv2.rectangle(frame, (right_x, 20), (right_x + box_width, 20 + box_height), (0, 0, 255), -1)
        cv2.putText(frame, f'Right Lane: {vehicles_right} vehicles', (right_x + 10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Status: {intensity_right}', (right_x + 10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.7, (255, 255, 255), 2)
    
    def get_statistics(self):
        """Get analysis statistics"""
        elapsed = time.time() - self.stats['analysis_start']
        fps = self.stats['total_frames'] / elapsed if elapsed > 0 else 0
        
        return {
            'total_frames': self.stats['total_frames'],
            'vehicles_detected': self.stats['vehicles_detected'],
            'elapsed_time': elapsed,
            'fps': fps,
            'vehicles_per_frame': self.stats['vehicles_detected'] / max(1, self.stats['total_frames'])
        }

def initialize_video_source(source):
    """Initialize video source with proper handling for different types"""
    print(f"\n🔌 Initializing video source: {source}")
    
    # Handle YouTube URLs
    if isinstance(source, str) and ('youtube.com' in source or 'youtu.be' in source):
        print("🎥 YouTube stream detected")
        return YouTubeStreamManager(source)
    
    # Handle other sources
    try:
        # Try different backends for better compatibility
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        
        for i, backend in enumerate(backends):
            print(f"🔄 Trying backend {i+1}/3...")
            cap = cv2.VideoCapture(source, backend)
            
            if cap.isOpened():
                # Test frame reading
                ret, frame = cap.read()
                if ret:
                    print(f"✅ Connected using backend {i+1}")
                    height, width = frame.shape[:2]
                    print(f"📐 Resolution: {width}x{height}")
                    
                    # Optimize settings
                    if isinstance(source, str) and ('rtsp://' in source or 'http://' in source):
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FPS, 30)
                        print("🎯 Optimized for live stream")
                    
                    return cap
                else:
                    cap.release()
            
        print("❌ All backends failed")
        return None
        
    except Exception as e:
        print(f"❌ Error initializing source: {e}")
        return None

def get_source_selection():
    """Interactive source selection"""
    print("\n=== INPUT SOURCE SELECTION ===")
    print("1. Video file")
    print("2. Webcam (default camera)")
    print("3. IP Camera/HTTP Stream")
    print("4. RTSP Stream")
    print("5. YouTube Live Stream")
    
    choice = input("Select input source (1-5): ").strip()
    
    if choice == "1":
        path = input("Enter video file path (or press Enter for default): ").strip().strip('"')
        return path if path else "videos/indian_traffic.mp4"
    elif choice == "2":
        cam_id = input("Enter camera ID (default 0): ").strip()
        return int(cam_id) if cam_id.isdigit() else 0
    elif choice == "3":
        url = input("Enter IP camera URL: ").strip()
        return url
    elif choice == "4":
        rtsp_url = input("Enter RTSP URL: ").strip()
        return rtsp_url
    elif choice == "5":
        youtube_url = input("Enter YouTube live stream URL: ").strip()
        return youtube_url if youtube_url else "https://www.youtube.com/live/_IFD0Ah8a-M?si=XlnzhwXGBJy_eIWW"
    else:
        print("Invalid choice. Using default video.")
        return "videos/indian_traffic.mp4"

def load_default_config():
    """Load default lane configuration"""
    return {
        'polygons': [
            [(465, 350), (609, 350), (510, 630), (2, 630)],
            [(678, 350), (815, 350), (1203, 630), (743, 630)]
        ],
        'lane_threshold': 609,
        'detection_area': (325, 635)
    }

def load_saved_config():
    """Load saved lane configuration"""
    try:
        with open('lane_config.json', 'r') as f:
            data = json.load(f)
        print("✅ Loaded saved lane configuration")
        return {
            'polygons': data['polygons'],
            'lane_threshold': data['lane_threshold'],
            'detection_area': data['detection_area']
        }
    except:
        print("⚠️  No saved configuration found, using default")
        return load_default_config()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Real-time Vehicle Detection and Traffic Analysis')
    parser.add_argument('--source', type=str, default=None, 
                       help='Video source (file, webcam, URL, YouTube)')
    parser.add_argument('--weights', type=str, default='models/best.pt', 
                       help='Path to YOLO model weights')
    parser.add_argument('--conf', type=float, default=0.4, 
                       help='Detection confidence threshold')
    parser.add_argument('--no-calibration', action='store_true', 
                       help='Skip calibration, use default/saved polygons')
    parser.add_argument('--headless', action='store_true', 
                       help='Run without GUI (for servers/automation)')
    parser.add_argument('--max-frames', type=int, default=None, 
                       help='Maximum frames to process (for testing)')
    parser.add_argument('--output', type=str, default=None, 
                       help='Save processed video to file')
    
    args = parser.parse_args()
    
    print("🚗 Real-Time Vehicle Detection and Traffic Analysis")
    print("=" * 60)
    
    # Check model file
    if not os.path.exists(args.weights):
        print(f"❌ Model file '{args.weights}' not found!")
        return
    
    # Load YOLO model
    try:
        model = YOLO(args.weights)
        print(f"✅ Model loaded: {args.weights}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Get video source
    if args.source is None:
        source = get_source_selection()
    else:
        source = args.source
    
    print(f"\n📹 Video source: {source}")
    
    # Initialize video source
    cap = initialize_video_source(source)
    if cap is None:
        print("❌ Failed to initialize video source")
        return
    
    # Get sample frame for calibration
    if isinstance(cap, YouTubeStreamManager):
        ret, sample_frame = cap.read()
    else:
        ret, sample_frame = cap.read()
    
    if not ret:
        print("❌ Could not read sample frame")
        if hasattr(cap, 'release'):
            cap.release()
        return
    
    print(f"📐 Frame size: {sample_frame.shape[1]}x{sample_frame.shape[0]}")
    
    # Configure lanes
    if args.no_calibration:
        print("\n🎯 Using saved/default lane configuration")
        config = load_saved_config()
    else:
        if args.headless:
            print("\n🎯 Headless mode: using saved/default configuration")
            config = load_saved_config()
        else:
            print("\n🎯 Starting interactive calibration...")
            calibrator = StreamCalibrator(cap if not isinstance(cap, YouTubeStreamManager) else 
                                        cv2.VideoCapture(source))
            config = calibrator.calibrate_stream()
            
            if config is None:
                print("Calibration cancelled")
                if hasattr(cap, 'release'):
                    cap.release()
                return
    
    print(f"\n✅ Configuration loaded:")
    print(f"   Polygons: {len(config['polygons'])}")
    print(f"   Lane threshold: {config['lane_threshold']}")
    print(f"   Detection area: {config['detection_area']}")
    
    # Initialize traffic analyzer
    analyzer = TrafficAnalyzer(model, config, headless=args.headless)
    
    # Setup video output if requested
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = 20
        width, height = sample_frame.shape[1], sample_frame.shape[0]
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"📹 Output video: {args.output}")
    
    # Setup display window (if not headless)
    if not args.headless:
        window_name = 'Traffic Analysis'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, 1200, 675)
        cv2.moveWindow(window_name, 50, 50)
    
    print(f"\n🚀 Starting traffic analysis...")
    print(f"Mode: {'Headless' if args.headless else 'GUI'}")
    if not args.headless:
        print("Controls: 'q' to quit, 's' to save frame, 'p' to pause")
    
    # Main processing loop
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    paused = False
    
    try:
        while True:
            # Read frame
            if isinstance(cap, YouTubeStreamManager):
                ret, frame = cap.read()
            else:
                ret, frame = cap.read()
            
            if not ret:
                if isinstance(cap, YouTubeStreamManager):
                    print("⚠️  YouTube stream read failed, retrying...")
                    time.sleep(0.1)
                    continue
                else:
                    print("⚠️  End of video or read failed")
                    break
            
            frame_count += 1
            fps_counter += 1
            
            # Check frame limit
            if args.max_frames and frame_count >= args.max_frames:
                print(f"✅ Reached maximum frames ({args.max_frames})")
                break
            
            # Process frame
            if not paused:
                processed_frame, vehicles_left, vehicles_right = analyzer.analyze_frame(frame)
                
                # Calculate FPS
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    current_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_start = time.time()
                    
                    # Add FPS to frame
                    if not args.headless:
                        cv2.putText(processed_frame, f'FPS: {current_fps:.1f}', 
                                   (10, processed_frame.shape[0] - 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save to output video
                if out_writer:
                    out_writer.write(processed_frame)
                
                # Display frame (if not headless)
                if not args.headless:
                    display_frame = cv2.resize(processed_frame, (1200, 675))
                    cv2.imshow(window_name, display_frame)
                
                # Print progress (headless mode)
                if args.headless and frame_count % 30 == 0:
                    stats = analyzer.get_statistics()
                    print(f"Frame {frame_count}: L={vehicles_left}, R={vehicles_right}, "
                          f"FPS={stats['fps']:.1f}, Total vehicles={stats['vehicles_detected']}")
            
            # Handle keyboard input (if not headless)
            if not args.headless:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"traffic_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"📸 Frame saved: {filename}")
            
    except KeyboardInterrupt:
        print("\n⏹️  Analysis stopped by user")
    
    finally:
        # Cleanup
        if hasattr(cap, 'release'):
            cap.release()
        if out_writer:
            out_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        
        # Print final statistics
        stats = analyzer.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Total frames processed: {stats['total_frames']}")
        print(f"   Total vehicles detected: {stats['vehicles_detected']}")
        print(f"   Average FPS: {stats['fps']:.2f}")
        print(f"   Vehicles per frame: {stats['vehicles_per_frame']:.2f}")
        print(f"   Analysis duration: {stats['elapsed_time']:.2f} seconds")
        
        print(f"\n✅ Traffic analysis completed!")

if __name__ == "__main__":
    main()
