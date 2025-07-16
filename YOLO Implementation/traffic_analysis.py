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
import traceback
import asyncio
from queue import Queue
from ultralytics import YOLO
import json
from datetime import datetime

# Import our custom modules
from config import config
from violence_detector import create_violence_detector, create_alert_manager
from animal_detector import AnimalDetector, AnimalAlertManager

# === SUMO Traffic Control Integration ===
try:
    import traci
    SUMO_AVAILABLE = True
    print("✅ SUMO TraCI available - Traffic control enabled")
except ImportError:
    SUMO_AVAILABLE = False
    print("⚠️ SUMO TraCI not available - Traffic control disabled")

# === SMS Alert Integration ===
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
    print("✅ Twilio available - SMS alerts enabled")
except ImportError:
    TWILIO_AVAILABLE = False
    print("⚠️ Twilio not available - SMS alerts disabled")

# === SUMO Configuration ===
SUMO_CONFIG_PATH = "../simulation_files/map.sumocfg"  # Fixed path to parent directory
JUNCTION_ID = "n1"

# === Lane-Specific Traffic Control Parameters ===
LEFT_LANE_THRESHOLD = 8   # Vehicles in left lane to trigger control
RIGHT_LANE_THRESHOLD = 8  # Vehicles in right lane to trigger control
DENSITY_IMBALANCE_RATIO = 2.0  # Ratio difference to trigger lane priority
EMERGENCY_STOP_DURATION = 30   # seconds to stop traffic for emergencies

# === Alert Duration Parameters ===
VIOLENCE_ALERT_DURATION = 100  # seconds
ANIMAL_ALERT_DURATION = 20     # seconds

# === Traffic Light States for 2-Lane Intersection ===
TRAFFIC_STATES = {
    'NORMAL_FLOW': "GGrGG",      # Both lanes green (normal operation)
    'LEFT_PRIORITY': "GGrrr",    # Prioritize left lane 
    'RIGHT_PRIORITY': "rrGGG",   # Prioritize right lane
    'EMERGENCY_STOP': "rrrrr",   # Emergency stop all traffic
    'ANIMAL_CAUTION': "yyyyy"    # Yellow caution for animal presence
}

# === Twilio Configuration (can be overridden via config) ===
TWILIO_ACCOUNT_SID = "AC3e9621790fea1070b200fd40a16c1191"
TWILIO_AUTH_TOKEN = "d25bc282410c2247ac6f71f2b55c1cdf"
TWILIO_PHONE_NUMBER = "+16284710074"
POLICE_NUMBER = "+919877035742"  # Default emergency number

# === Traffic Control State Variables ===
violence_detected_start = None
violence_alert_sent = False
animal_detected_start = None
animal_light_adjusted = False

# === SUMO Traffic Control Functions ===
def start_sumo():
    """Initialize SUMO traffic simulation"""
    if not SUMO_AVAILABLE:
        print("⚠️ SUMO not available - Traffic control simulation disabled")
        return False
    
    try:
        if not os.path.exists(SUMO_CONFIG_PATH):
            print(f"⚠️ SUMO config file not found: {SUMO_CONFIG_PATH}")
            return False
            
        traci.start(["sumo-gui", "-c", SUMO_CONFIG_PATH])
        print("✅ SUMO traffic simulation started")
        return True
    except Exception as e:
        print(f"❌ Failed to start SUMO: {e}")
        return False

def stop_sumo():
    """Stop SUMO traffic simulation"""
    if SUMO_AVAILABLE and traci.isLoaded():
        try:
            traci.close()
            print("🛑 SUMO traffic simulation stopped")
        except Exception as e:
            print(f"⚠️ Error stopping SUMO: {e}")

def control_traffic_light(vehicles_left, vehicles_right):
    """Enhanced lane-specific traffic control based on individual lane density"""
    if not SUMO_AVAILABLE or not traci.isLoaded():
        return {"action": "SUMO_DISABLED", "reason": "SUMO not available"}
        
    try:
        # Analyze lane-specific traffic conditions
        left_heavy = vehicles_left >= LEFT_LANE_THRESHOLD
        right_heavy = vehicles_right >= RIGHT_LANE_THRESHOLD
        total_vehicles = vehicles_left + vehicles_right
        
        # Calculate density imbalance ratio
        if vehicles_right > 0:
            density_ratio = vehicles_left / vehicles_right
        else:
            density_ratio = float('inf') if vehicles_left > 0 else 1.0
            
        # Determine traffic control action based on lane analysis
        if violence_detected_start is not None:
            # Emergency: Stop all traffic for violence
            current_state = TRAFFIC_STATES['EMERGENCY_STOP']
            action_reason = f"🚨 EMERGENCY STOP - Violence detected"
            action_type = "EMERGENCY_VIOLENCE"
            
        elif animal_detected_start is not None:
            # Caution: Yellow lights for animal presence  
            current_state = TRAFFIC_STATES['ANIMAL_CAUTION']
            action_reason = f"🐕 ANIMAL CAUTION - {total_vehicles} vehicles detected"
            action_type = "ANIMAL_CAUTION"
            
        elif left_heavy and right_heavy:
            # Both lanes congested - normal flow to prevent deadlock
            current_state = TRAFFIC_STATES['NORMAL_FLOW']
            action_reason = f"🚦 BOTH LANES HEAVY (L:{vehicles_left}, R:{vehicles_right}) - Normal flow"
            action_type = "BOTH_LANES_HEAVY"
            
        elif left_heavy and not right_heavy:
            # Left lane congested - prioritize left lane
            current_state = TRAFFIC_STATES['LEFT_PRIORITY']
            action_reason = f"⬅️ LEFT LANE PRIORITY ({vehicles_left} vehicles) - Right clear ({vehicles_right})"
            action_type = "LEFT_PRIORITY"
            
        elif right_heavy and not left_heavy:
            # Right lane congested - prioritize right lane
            current_state = TRAFFIC_STATES['RIGHT_PRIORITY']
            action_reason = f"➡️ RIGHT LANE PRIORITY ({vehicles_right} vehicles) - Left clear ({vehicles_left})"
            action_type = "RIGHT_PRIORITY"
            
        elif density_ratio >= DENSITY_IMBALANCE_RATIO:
            # Significant left lane imbalance
            current_state = TRAFFIC_STATES['LEFT_PRIORITY']
            action_reason = f"⬅️ LEFT DENSITY IMBALANCE (ratio: {density_ratio:.1f}) - L:{vehicles_left}, R:{vehicles_right}"
            action_type = "DENSITY_IMBALANCE_LEFT"
            
        elif density_ratio <= (1.0 / DENSITY_IMBALANCE_RATIO):
            # Significant right lane imbalance  
            current_state = TRAFFIC_STATES['RIGHT_PRIORITY']
            action_reason = f"➡️ RIGHT DENSITY IMBALANCE (ratio: {density_ratio:.1f}) - L:{vehicles_left}, R:{vehicles_right}"
            action_type = "DENSITY_IMBALANCE_RIGHT"
            
        else:
            # Normal traffic conditions
            current_state = TRAFFIC_STATES['NORMAL_FLOW']
            action_reason = f"✅ NORMAL FLOW - Balanced traffic (L:{vehicles_left}, R:{vehicles_right})"
            action_type = "NORMAL_FLOW"
        
        # Apply traffic light state
        traci.trafficlight.setRedYellowGreenState(JUNCTION_ID, current_state)
        
        # Get current phase for confirmation
        current_phase = traci.trafficlight.getPhase(JUNCTION_ID)
        
        print(f"🚦 Traffic Control Applied:")
        print(f"   {action_reason}")
        print(f"   Signal State: {current_state} (Phase: {current_phase})")
        
        return {
            "action": action_type,
            "reason": action_reason,
            "vehicles_left": vehicles_left,
            "vehicles_right": vehicles_right, 
            "traffic_state": current_state,
            "phase": current_phase,
            "density_ratio": density_ratio
        }
            
    except Exception as e:
        error_msg = f"❌ Error controlling traffic light: {e}"
        print(error_msg)
        return {"action": "ERROR", "reason": error_msg}

def adjust_traffic_light_for_animals():
    """Extend traffic light timing when animals are detected"""
    if not SUMO_AVAILABLE or not traci.isLoaded():
        return
        
    try:
        print("🦘 Animal detected >20s: Extending green phase for safety")
        current_phase = traci.trafficlight.getPhase(JUNCTION_ID)
        traci.trafficlight.setPhaseDuration(JUNCTION_ID, 60)  # Extend to 60 seconds
        print(f"✅ Traffic phase {current_phase} extended to 60s for animal safety")
    except Exception as e:
        print(f"❌ Failed to adjust traffic light for animals: {e}")

# === SMS Alert Functions ===
def send_sms_alert(phone_number, message):
    """Send SMS alert using Twilio"""
    if not TWILIO_AVAILABLE:
        print(f"⚠️ SMS Alert (Twilio disabled): {message}")
        return False
        
    try:
        print(f"📲 Sending emergency SMS to {phone_number}")
        print(f"   Message: {message}")
        
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        sms_message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        print(f"✅ Emergency SMS sent successfully. SID: {sms_message.sid}")
        return True
    except Exception as e:
        print(f"❌ Failed to send SMS alert: {e}")
        return False

def check_violence_alert(current_violence_alert):
    """Monitor violence detection and send alerts when threshold exceeded"""
    global violence_detected_start, violence_alert_sent
    
    if current_violence_alert:
        if violence_detected_start is None:
            violence_detected_start = time.time()
            print(f"🚨 Violence monitoring started - Alert threshold: {VIOLENCE_ALERT_DURATION}s")
        
        elapsed = time.time() - violence_detected_start
        if elapsed > VIOLENCE_ALERT_DURATION and not violence_alert_sent:
            alert_message = f"EMERGENCY: Violence detected continuously for {elapsed:.0f} seconds at traffic intersection. Immediate police response required."
            send_sms_alert(POLICE_NUMBER, alert_message)
            violence_alert_sent = True
            print(f"🚨 CRITICAL: Violence alert sent to authorities after {elapsed:.0f}s")
    else:
        if violence_detected_start is not None:
            elapsed = time.time() - violence_detected_start
            print(f"✅ Violence monitoring ended after {elapsed:.0f}s")
        violence_detected_start = None
        violence_alert_sent = False

def check_animal_duration(current_animal_alert):
    """Monitor animal detection and adjust traffic accordingly"""
    global animal_detected_start, animal_light_adjusted
    
    if current_animal_alert:
        if animal_detected_start is None:
            animal_detected_start = time.time()
            print(f"🐕 Animal monitoring started - Traffic adjustment threshold: {ANIMAL_ALERT_DURATION}s")
        
        elapsed = time.time() - animal_detected_start
        if elapsed > ANIMAL_ALERT_DURATION and not animal_light_adjusted:
            adjust_traffic_light_for_animals()
            animal_light_adjusted = True
            print(f"🚦 Traffic lights adjusted for animal safety after {elapsed:.0f}s")
    else:
        if animal_detected_start is not None:
            elapsed = time.time() - animal_detected_start
            print(f"✅ Animal monitoring ended after {elapsed:.0f}s")
        animal_detected_start = None
        animal_light_adjusted = False

def process_traffic_control_frame(vehicles_left, vehicles_right, current_violence_alert, current_animal_alert):
    """Process one frame for traffic control decisions with detailed lane analysis"""
    control_result = {"action": "DISABLED", "reason": "Traffic control disabled"}
    
    try:
        # Step SUMO simulation
        if SUMO_AVAILABLE and traci.isLoaded():
            traci.simulationStep()
        
        # Control traffic based on lane-specific vehicle density
        control_result = control_traffic_light(vehicles_left, vehicles_right)
        
        # Monitor violence detection
        check_violence_alert(current_violence_alert)
        
        # Monitor animal detection  
        check_animal_duration(current_animal_alert)
        
        return control_result
        
    except Exception as e:
        error_msg = f"⚠️ Error in traffic control processing: {e}"
        print(error_msg)
        return {"action": "ERROR", "reason": error_msg}

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
                self.consecutive_failures > 1)  # Refresh sooner on failures
    
    def connect(self):
        """Connect with fresh URL if needed"""
        if self.current_stream_url is None or self.should_refresh_url():
            self.current_stream_url = self.get_fresh_stream_url()
            
            if not self.current_stream_url:
                return False
        
        try:
            if self.cap:
                self.cap.release()
                
            # Use more robust settings for YouTube streams
            self.cap = cv2.VideoCapture(self.current_stream_url, cv2.CAP_FFMPEG)
            
            if self.cap.isOpened():
                # Optimize settings for live streams with better buffering
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Slightly larger buffer for stability
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)  # Longer timeout
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 8000)   # Longer read timeout
                    print("   ✅ Basic stream settings applied")
                except Exception as e:
                    print(f"   ⚠️ Could not apply some stream settings: {e}")
                
                # Additional stability settings (if supported by OpenCV version)
                try:
                    if hasattr(cv2, 'CAP_PROP_RECONNECT_THRESHOLD'):
                        self.cap.set(cv2.CAP_PROP_RECONNECT_THRESHOLD, 1000)
                        self.cap.set(cv2.CAP_PROP_RECONNECT_DELAY_MAX, 5000)
                        print("   ✅ Advanced reconnection settings applied")
                    else:
                        print("   ℹ️ Advanced reconnection settings not available (older OpenCV)")
                except Exception as e:
                    print(f"   ⚠️ Advanced reconnection settings failed: {e}")
                
                # Test read with retry
                for attempt in range(3):
                    ret, frame = self.cap.read()
                    if ret:
                        print(f"✅ YouTube stream connected successfully (attempt {attempt + 1})")
                        self.consecutive_failures = 0
                        return True
                    print(f"   ❌ Connection attempt {attempt + 1} failed")
                    time.sleep(1)
                    
            print("❌ Failed to connect to YouTube stream")
            self.consecutive_failures += 1
            return False
            
        except Exception as e:
            print(f"❌ YouTube connection error: {e}")
            self.consecutive_failures += 1
            return False
    
    def read(self):
        """Read frame with auto-reconnection and better error handling"""
        if not self.cap or not self.cap.isOpened():
            if not self.connect():
                return False, None
        
        # Try to read frame with timeout handling
        try:
            ret, frame = self.cap.read()
            
            if not ret:
                self.consecutive_failures += 1
                print(f"⚠️ Frame read failed (attempt {self.consecutive_failures}/{self.max_failures})")
                
                if self.consecutive_failures < self.max_failures:
                    # Force URL refresh for connection issues
                    if self.consecutive_failures >= 2:
                        print("🔄 Forcing stream URL refresh due to multiple failures...")
                        self.current_stream_url = None
                        self.last_refresh = 0
                    
                    print("🔄 Attempting reconnection...")
                    time.sleep(2)  # Longer delay for stability
                    
                    if self.connect():
                        ret, frame = self.cap.read()
                        if ret:
                            print("✅ Reconnection successful")
                            self.consecutive_failures = 0
                            return ret, frame
                else:
                    print("❌ Max YouTube failures reached, stream may be unstable")
                    return False, None
            else:
                # Successful read, reset failure counter
                if self.consecutive_failures > 0:
                    print(f"✅ Stream recovered after {self.consecutive_failures} failures")
                    self.consecutive_failures = 0
                return ret, frame
                
        except Exception as e:
            self.consecutive_failures += 1
            print(f"❌ Read exception: {e}")
            if self.consecutive_failures < self.max_failures:
                time.sleep(1)
                return self.read()  # Retry once
            
        return False, None
    
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
        
        # Get video FPS for proper playback timing
        video_fps = 30  # Default FPS
        try:
            if hasattr(self.cap, 'get') and not isinstance(self.cap, YouTubeStreamManager):
                detected_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if detected_fps > 0:
                    video_fps = detected_fps
                    print(f"📹 Video FPS detected: {video_fps:.1f}")
                else:
                    print("📹 Could not detect FPS, using default: 30")
            else:
                print("📹 Using default FPS for live stream: 30")
        except:
            print("📹 FPS detection failed, using default: 30")
        
        # Calculate frame delay for natural playback speed
        frame_delay = 1.0 / video_fps  # Seconds per frame
        frame_delay_ms = max(1, int(frame_delay * 1000))  # Convert to milliseconds, minimum 1ms
        
        print(f"🎬 Calibration playback: {video_fps:.1f} FPS (delay: {frame_delay_ms}ms per frame)")
        print("   Video will play at normal speed during calibration")
        
        # Initialize frame counter for calibration
        calibration_frame_count = 0
        calibration_start_time = time.time()
        
        while True:
            frame_start_time = time.time()
            calibration_frame_count += 1
            
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read from stream!")
                return None
                
            self.original_frame = frame.copy()
            self.draw_polygons()
            
            # Calculate elapsed time and current playback info
            elapsed_time = time.time() - calibration_start_time
            current_fps = calibration_frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add instructions overlay
            cv2.putText(self.frame, f"Polygons: {len(self.polygons)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(self.frame, "Press 'c' to complete, 'r' to reset", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(self.frame, f"Frame: {calibration_frame_count} | FPS: {current_fps:.1f}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(self.frame, f"Time: {elapsed_time:.1f}s", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Scale frame for display if needed
            if self.display_scale != 1.0:
                display_frame = cv2.resize(self.frame, (self.display_width, self.display_height))
            else:
                display_frame = self.frame
            
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard input with proper timing
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
            
            # Control frame rate for natural video playback
            frame_elapsed = time.time() - frame_start_time
            sleep_time = frame_delay - frame_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
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
    """Main traffic analysis processor with integrated violence detection and traffic control"""
    
    def __init__(self, model, lane_config, headless=False, enable_traffic_control=True):
        self.model = model
        self.lane_config = lane_config
        self.headless = headless
        self.enable_traffic_control = enable_traffic_control
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=5)
        self.processing = False
        self.stats = {
            'total_frames': 0,
            'vehicles_detected': 0,
            'analysis_start': time.time(),
            'traffic_control_actions': 0,
            'violence_alerts_sent': 0,
            'animal_traffic_adjustments': 0,
            'left_lane_control_actions': 0,
            'right_lane_control_actions': 0,
            'emergency_stops': 0
        }
        
        # Traffic control state tracking
        self.current_traffic_control = {
            "action": "INITIALIZING",
            "reason": "System starting up",
            "vehicles_left": 0,
            "vehicles_right": 0,
            "traffic_state": "NORMAL_FLOW",
            "density_ratio": 1.0
        }
        
        # Initialize traffic control system
        self.sumo_enabled = False
        if self.enable_traffic_control and SUMO_AVAILABLE:
            self.sumo_enabled = start_sumo()
            if self.sumo_enabled:
                print("🚦 [TRAFFIC CONTROL] SUMO traffic simulation integrated")
            else:
                print("⚠️ [TRAFFIC CONTROL] SUMO simulation failed to start")
        else:
            print("⚠️ [TRAFFIC CONTROL] Traffic control disabled or SUMO unavailable")
        
        # Initialize violence detection
        self.violence_detector = create_violence_detector(config)
        self.alert_manager = create_alert_manager(config)
        self.current_violence_alert = None
        self.alert_display_time = 0
        self.alert_duration = 5.0  # seconds
        
        if self.violence_detector:
            self.violence_detector.start_processing()
            print("[SHIELD] Violence detection enabled and started")
        else:
            print("[WARNING] Violence detection disabled")
        
        # Initialize animal detection
        self.animal_detector = AnimalDetector()
        self.animal_alert_manager = AnimalAlertManager()
        self.animal_detection_interval = 5  # Check every 5 frames (was 30 - too infrequent!)
        self.animal_frame_counter = 0
        self.current_animal_alert = None
        self.animal_alert_display_time = 0
        self.animal_alert_duration = 5.0  # seconds
        
        print("[ANIMAL] Animal detection initialized and enabled")
        print(f"[ANIMAL] Detection interval: every {self.animal_detection_interval} frames")
        
    def analyze_frame(self, frame):
        """Analyze single frame for vehicle detection, violence detection, and animal detection"""
        self.stats['total_frames'] += 1
        self.animal_frame_counter += 1
        
        # Queue frame for violence detection (async)
        if self.violence_detector:
            self.violence_detector.queue_frame(frame)
        
        # Process any pending violence detection results
        self._process_violence_results()
        
        # Process animal detection every N frames
        if self.animal_frame_counter >= self.animal_detection_interval:
            self._process_animal_detection(frame)
            self.animal_frame_counter = 0
        
        # Process any pending animal detection results
        self._process_animal_results()
        
        detection_frame = frame.copy()
        x1, x2 = self.lane_config['detection_area']
        
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
        for i, poly in enumerate(self.lane_config['polygons']):
            pts = np.array(poly, dtype=np.int32)
            color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
            cv2.polylines(processed_frame, [pts], True, color, 2)
        
        # Count vehicles in each lane
        vehicles_left = 0
        vehicles_right = 0
        lane_threshold = self.lane_config['lane_threshold']
        
        if results[0].boxes is not None:
            total_vehicles = len(results[0].boxes)
            self.stats['vehicles_detected'] += total_vehicles
            
            for box in results[0].boxes.xyxy:
                if box[0] < lane_threshold:
                    vehicles_left += 1
                else:
                    vehicles_right += 1
        
        # === TRAFFIC CONTROL INTEGRATION ===
        # Process traffic control decisions based on current detections
        if self.enable_traffic_control:
            try:
                control_result = process_traffic_control_frame(
                    vehicles_left, 
                    vehicles_right, 
                    self.current_violence_alert, 
                    self.current_animal_alert
                )
                
                # Store control result for display
                self.current_traffic_control = control_result
                self.stats['traffic_control_actions'] += 1
                
                # Update specific action counters
                if control_result.get('action') == 'LEFT_PRIORITY':
                    self.stats['left_lane_control_actions'] += 1
                elif control_result.get('action') == 'RIGHT_PRIORITY':
                    self.stats['right_lane_control_actions'] += 1
                elif control_result.get('action') in ['EMERGENCY_VIOLENCE', 'ANIMAL_CAUTION']:
                    self.stats['emergency_stops'] += 1
                    
            except Exception as e:
                print(f"⚠️ Traffic control error: {e}")
                self.current_traffic_control = {
                    "action": "ERROR", 
                    "reason": str(e),
                    "vehicles_left": vehicles_left,
                    "vehicles_right": vehicles_right
                }
        
        # Add traffic analysis annotations
        self.add_annotations(processed_frame, vehicles_left, vehicles_right)
        
        # Add violence detection overlay if active alert
        if self.current_violence_alert and not self.headless:
            processed_frame = self.violence_detector.create_alert_overlay(
                processed_frame, self.current_violence_alert
            )
        
        # Add animal detection overlay if active alert
        if self.current_animal_alert and not self.headless:
            processed_frame = self._create_animal_alert_overlay(
                processed_frame, self.current_animal_alert
            )
        
        return processed_frame, vehicles_left, vehicles_right
    
    def add_annotations(self, frame, vehicles_left, vehicles_right):
        """Add traffic analysis overlays with resolution-adaptive scaling and crisp rendering"""
        heavy_threshold = 8
        
        # Traffic intensity
        intensity_left = "HEAVY TRAFFIC" if vehicles_left > heavy_threshold else "NORMAL FLOW"
        intensity_right = "HEAVY TRAFFIC" if vehicles_right > heavy_threshold else "NORMAL FLOW"
        
        frame_height, frame_width = frame.shape[:2]
        
        # Advanced resolution-adaptive scaling for crisp rendering
        # Calculate pixel density factor for better scaling
        pixel_density = (frame_width * frame_height) / (1920 * 1080)  # Reference: 1080p
        density_factor = min(max(pixel_density, 0.3), 2.5)  # Clamp between 0.3x and 2.5x
        
        # Multi-tier font scaling for different resolutions
        if frame_width >= 3840:  # 4K and above
            base_font_scale = 2.2
            thickness_multiplier = 4
        elif frame_width >= 2560:  # 2K
            base_font_scale = 1.8
            thickness_multiplier = 3
        elif frame_width >= 1920:  # 1080p
            base_font_scale = 1.4
            thickness_multiplier = 3
        elif frame_width >= 1280:  # 720p
            base_font_scale = 1.0
            thickness_multiplier = 2
        else:  # Lower resolutions
            base_font_scale = 0.8
            thickness_multiplier = 2
        
        # Apply density factor for smooth scaling
        font_scale = base_font_scale * density_factor
        font_scale = max(0.6, min(font_scale, 3.0))  # Reasonable bounds
        
        # Resolution-adaptive dimensions with integer precision
        box_width = max(int(frame_width * 0.28), 300)  # Minimum width
        box_height = max(int(120 * density_factor), 100)  # Minimum height
        border_thickness = max(int(3 * density_factor), 2)  # Minimum border
        
        # Text thickness for crisp rendering
        text_thickness = max(int(thickness_multiplier * density_factor), 2)
        
        # Determine left lane traffic light status
        if self.enable_traffic_control and hasattr(self, 'current_traffic_control'):
            traffic_action = self.current_traffic_control.get('action', 'NORMAL_FLOW')
            if traffic_action in ['LEFT_PRIORITY', 'NORMAL_FLOW']:
                left_light_color = (0, 255, 0)  # Green
                left_light_text = "GREEN LIGHT"
                left_bg_color = (0, 120, 0)  # Dark green background
            elif traffic_action in ['EMERGENCY_VIOLENCE', 'ANIMAL_CAUTION']:
                left_light_color = (255, 255, 255)  # White text
                left_light_text = "EMERGENCY STOP"
                left_bg_color = (0, 0, 180)  # Dark red background
            elif traffic_action == 'RIGHT_PRIORITY':
                left_light_color = (255, 255, 255)  # White text
                left_light_text = "RED LIGHT"
                left_bg_color = (0, 0, 180)  # Dark red background
            else:
                left_light_color = (0, 0, 0)  # Black text
                left_light_text = "CAUTION"
                left_bg_color = (0, 220, 220)  # Yellow background
        else:
            left_light_color = (220, 220, 220)  # Light gray
            left_light_text = "SYSTEM DISABLED"
            left_bg_color = (100, 100, 100)  # Dark gray background
        
        # Left lane info box with pixel-perfect positioning
        shadow_offset = max(int(5 * density_factor), 3)
        
        # Background shadow with anti-aliasing
        cv2.rectangle(frame, (15, 15), (15 + box_width + shadow_offset, 15 + box_height + shadow_offset), (0, 0, 0), -1)
        # Main rectangle with precise edges
        cv2.rectangle(frame, (20, 20), (20 + box_width, 20 + box_height), left_bg_color, -1)
        # Crisp white border
        cv2.rectangle(frame, (20, 20), (20 + box_width, 20 + box_height), (255, 255, 255), border_thickness)
        
        # Left lane text with anti-aliased rendering and precise positioning
        text_margin = max(int(15 * density_factor), 10)
        line_spacing = max(int(25 * density_factor), 20)
        
        cv2.putText(frame, 'LEFT LANE', (20 + text_margin, 20 + line_spacing), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.85, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, f'Vehicles: {vehicles_left}', (20 + text_margin, 20 + line_spacing * 2), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.7, (255, 255, 255), text_thickness - 1, cv2.LINE_AA)
        cv2.putText(frame, f'{intensity_left}', (20 + text_margin, 20 + line_spacing * 3), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.65, (255, 255, 255), text_thickness - 1, cv2.LINE_AA)
        cv2.putText(frame, f'{left_light_text}', (20 + text_margin, 20 + line_spacing * 4), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.65, left_light_color, text_thickness - 1, cv2.LINE_AA)
        
        # Right lane info box with pixel-perfect positioning
        right_x = frame_width - box_width - 25
        
        # Determine right lane traffic light status
        if self.enable_traffic_control and hasattr(self, 'current_traffic_control'):
            traffic_action = self.current_traffic_control.get('action', 'NORMAL_FLOW')
            if traffic_action in ['RIGHT_PRIORITY', 'NORMAL_FLOW']:
                right_light_color = (0, 255, 0)  # Green
                right_light_text = "GREEN LIGHT"
                right_bg_color = (0, 120, 0)  # Dark green background
            elif traffic_action in ['EMERGENCY_VIOLENCE', 'ANIMAL_CAUTION']:
                right_light_color = (255, 255, 255)  # White text
                right_light_text = "EMERGENCY STOP"
                right_bg_color = (0, 0, 180)  # Dark red background
            elif traffic_action == 'LEFT_PRIORITY':
                right_light_color = (255, 255, 255)  # White text
                right_light_text = "RED LIGHT"
                right_bg_color = (0, 0, 180)  # Dark red background
            else:
                right_light_color = (0, 0, 0)  # Black text
                right_light_text = "CAUTION"
                right_bg_color = (0, 220, 220)  # Yellow background
        else:
            right_light_color = (220, 220, 220)  # Light gray
            right_light_text = "SYSTEM DISABLED"
            right_bg_color = (100, 100, 100)  # Dark gray background
        
        # Background shadow with anti-aliasing
        cv2.rectangle(frame, (right_x - shadow_offset, 15), (right_x + box_width + shadow_offset, 15 + box_height + shadow_offset), (0, 0, 0), -1)
        # Main rectangle with precise edges
        cv2.rectangle(frame, (right_x, 20), (right_x + box_width, 20 + box_height), right_bg_color, -1)
        # Crisp white border
        cv2.rectangle(frame, (right_x, 20), (right_x + box_width, 20 + box_height), (255, 255, 255), border_thickness)
        
        # Right lane text with anti-aliased rendering and precise positioning
        cv2.putText(frame, 'RIGHT LANE', (right_x + text_margin, 20 + line_spacing), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.85, (255, 255, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(frame, f'Vehicles: {vehicles_right}', (right_x + text_margin, 20 + line_spacing * 2), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.7, (255, 255, 255), text_thickness - 1, cv2.LINE_AA)
        cv2.putText(frame, f'{intensity_right}', (right_x + text_margin, 20 + line_spacing * 3), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.65, (255, 255, 255), text_thickness - 1, cv2.LINE_AA)
        cv2.putText(frame, f'{right_light_text}', (right_x + text_margin, 20 + line_spacing * 4), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.65, right_light_color, text_thickness - 1, cv2.LINE_AA)
        
        # Violence detection status box (bottom left) with crisp rendering
        if self.violence_detector and self.violence_detector.enabled:
            status_box_width = max(int(frame_width * 0.22), 280)
            status_box_height = max(int(90 * density_factor), 80)
            status_y = frame_height - status_box_height - max(int(25 * density_factor), 20)
            
            # Get violence detection statistics
            violence_stats = self.violence_detector.get_statistics()
            unack_alerts = self.alert_manager.get_unacknowledged_count()
            
            # Status color and text based on recent alerts
            if self.current_violence_alert:
                status_bg_color = (0, 0, 180)  # Dark red for active alert
                status_text = "VIOLENCE DETECTED"
                text_color = (255, 255, 255)
            elif unack_alerts > 0:
                status_bg_color = (0, 120, 220)  # Orange for pending alerts
                status_text = f"PENDING ALERTS: {unack_alerts}"
                text_color = (255, 255, 255)
            else:
                status_bg_color = (0, 120, 0)  # Dark green for normal
                status_text = "VIOLENCE MONITOR: OK"
                text_color = (255, 255, 255)
            
            # Background shadow with precise positioning
            cv2.rectangle(frame, (15, status_y - shadow_offset), (25 + status_box_width + shadow_offset, status_y + status_box_height + shadow_offset), (0, 0, 0), -1)
            # Main rectangle with crisp edges
            cv2.rectangle(frame, (20, status_y), (20 + status_box_width, status_y + status_box_height), status_bg_color, -1)
            # Anti-aliased border
            cv2.rectangle(frame, (20, status_y), (20 + status_box_width, status_y + status_box_height), (255, 255, 255), border_thickness)
            
            # Status box line spacing
            status_line_spacing = max(int(22 * density_factor), 18)
            
            cv2.putText(frame, 'SECURITY MONITOR', (20 + text_margin, status_y + status_line_spacing), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.7, text_color, text_thickness - 1, cv2.LINE_AA)
            cv2.putText(frame, status_text, (20 + text_margin, status_y + status_line_spacing * 2), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.6, text_color, text_thickness - 1, cv2.LINE_AA)
            cv2.putText(frame, 'AI-POWERED DETECTION', (20 + text_margin, status_y + status_line_spacing * 3), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.5, text_color, max(text_thickness - 2, 1), cv2.LINE_AA)
        
        # Animal detection status box (bottom right) with crisp rendering
        status_box_width = max(int(frame_width * 0.22), 280)
        status_box_height = max(int(90 * density_factor), 80)
        status_y = frame_height - status_box_height - max(int(25 * density_factor), 20)
        status_x = frame_width - status_box_width - 25
        
        # Get recent animal alerts
        recent_animal_alerts = self.animal_alert_manager.get_recent_alerts(hours=1)
        
        # Status color and text based on recent alerts and current detection
        if self.current_animal_alert:
            animal_bg_color = (0, 180, 180)  # Yellow-green for active detection
            animal_text = f"ANIMALS DETECTED: {self.current_animal_alert['total_animals']}"
            animal_text_color = (0, 0, 0)  # Black text on yellow background
        elif len(recent_animal_alerts) > 0:
            animal_bg_color = (0, 120, 220)  # Orange for recent alerts
            animal_text = f"RECENT ALERTS: {len(recent_animal_alerts)}"
            animal_text_color = (255, 255, 255)
        else:
            animal_bg_color = (0, 120, 0)  # Dark green for clear
            animal_text = "ANIMAL MONITOR: CLEAR"
            animal_text_color = (255, 255, 255)
        
        # Background shadow with precise positioning
        cv2.rectangle(frame, (status_x - shadow_offset, status_y - shadow_offset), (status_x + status_box_width + shadow_offset, status_y + status_box_height + shadow_offset), (0, 0, 0), -1)
        # Main rectangle with crisp edges
        cv2.rectangle(frame, (status_x, status_y), (status_x + status_box_width, status_y + status_box_height), animal_bg_color, -1)
        # Anti-aliased border
        cv2.rectangle(frame, (status_x, status_y), (status_x + status_box_width, status_y + status_box_height), (255, 255, 255), border_thickness)
        
        cv2.putText(frame, 'ANIMAL DETECTION', (status_x + text_margin, status_y + status_line_spacing), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.7, animal_text_color, text_thickness - 1, cv2.LINE_AA)
        cv2.putText(frame, animal_text, (status_x + text_margin, status_y + status_line_spacing * 2), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.6, animal_text_color, text_thickness - 1, cv2.LINE_AA)
        cv2.putText(frame, 'WILDLIFE SAFETY', (status_x + text_margin, status_y + status_line_spacing * 3), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.5, animal_text_color, max(text_thickness - 2, 1), cv2.LINE_AA)
        
        # Enhanced Traffic Control Status Box (center bottom) with pixel-perfect rendering
        if self.enable_traffic_control:
            traffic_box_width = max(int(frame_width * 0.4), 500)  # Minimum width for readability
            traffic_box_height = max(int(150 * density_factor), 120)  # Scale with resolution
            traffic_x = (frame_width - traffic_box_width) // 2
            traffic_y = frame_height - traffic_box_height - max(int(25 * density_factor), 20)
            
            # Get current traffic control status
            if hasattr(self, 'current_traffic_control'):
                control_action = self.current_traffic_control.get('action', 'INITIALIZING')
                control_reason = self.current_traffic_control.get('reason', 'System initializing')
                density_ratio = self.current_traffic_control.get('density_ratio', 1.0)
            else:
                control_action = "INITIALIZING"
                control_reason = "System starting up"
                density_ratio = 1.0
            
            # Determine traffic control status and color based on actual control action
            if control_action == "EMERGENCY_VIOLENCE":
                traffic_bg_color = (0, 0, 180)  # Dark red for emergency
                traffic_status = "EMERGENCY STOP - VIOLENCE"
                traffic_text_color = (255, 255, 255)
            elif control_action == "ANIMAL_CAUTION":
                traffic_bg_color = (0, 180, 180)  # Yellow for animal caution
                traffic_status = "ANIMAL CAUTION MODE"
                traffic_text_color = (0, 0, 0)
            elif control_action in ["LEFT_PRIORITY", "RIGHT_PRIORITY"]:
                traffic_bg_color = (0, 120, 220)  # Orange for active control
                traffic_status = f"ACTIVE: {control_action.replace('_', ' ')}"
                traffic_text_color = (255, 255, 255)
            elif control_action == "NORMAL_FLOW":
                traffic_bg_color = (0, 120, 0)  # Dark green for normal
                traffic_status = "NORMAL OPERATION"
                traffic_text_color = (255, 255, 255)
            elif not self.sumo_enabled:
                traffic_bg_color = (100, 100, 100)  # Gray for disabled
                traffic_status = "SUMO TRAFFIC CONTROL OFFLINE"
                traffic_text_color = (220, 220, 220)
            else:
                traffic_bg_color = (100, 100, 100)  # Gray for unknown
                traffic_status = f"STATUS: {control_action}"
                traffic_text_color = (220, 220, 220)
            
            # Background shadow with precise positioning
            cv2.rectangle(frame, (traffic_x - shadow_offset, traffic_y - shadow_offset), 
                         (traffic_x + traffic_box_width + shadow_offset, traffic_y + traffic_box_height + shadow_offset), 
                         (0, 0, 0), -1)
            # Main rectangle with crisp edges
            cv2.rectangle(frame, (traffic_x, traffic_y), 
                         (traffic_x + traffic_box_width, traffic_y + traffic_box_height), 
                         traffic_bg_color, -1)
            # Anti-aliased white border
            cv2.rectangle(frame, (traffic_x, traffic_y), 
                         (traffic_x + traffic_box_width, traffic_y + traffic_box_height), 
                         (255, 255, 255), border_thickness)
            
            # Enhanced text with precise spacing and anti-aliasing
            traffic_line_spacing = max(int(28 * density_factor), 22)
            traffic_text_margin = max(int(20 * density_factor), 15)
            
            cv2.putText(frame, 'TRAFFIC CONTROL SYSTEM', (traffic_x + traffic_text_margin, traffic_y + traffic_line_spacing), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.8, traffic_text_color, text_thickness, cv2.LINE_AA)
            cv2.putText(frame, f'Status: {traffic_status}', (traffic_x + traffic_text_margin, traffic_y + traffic_line_spacing * 2), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.7, traffic_text_color, text_thickness - 1, cv2.LINE_AA)
            
            # Lane threshold indicators with better formatting
            threshold_text = f'Thresholds: Left={LEFT_LANE_THRESHOLD}, Right={RIGHT_LANE_THRESHOLD}'
            cv2.putText(frame, threshold_text, (traffic_x + traffic_text_margin, traffic_y + traffic_line_spacing * 3), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.6, traffic_text_color, text_thickness - 1, cv2.LINE_AA)
            
            # Current vehicle counts and density ratio
            count_text = f'Current: L={vehicles_left}, R={vehicles_right}, Ratio={density_ratio:.1f}'
            cv2.putText(frame, count_text, (traffic_x + traffic_text_margin, traffic_y + traffic_line_spacing * 4), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.6, traffic_text_color, text_thickness - 1, cv2.LINE_AA)
            
            # System status indicators at bottom
            if violence_detected_start is not None:
                cv2.putText(frame, 'VIOLENCE ALERT ACTIVE', (traffic_x + traffic_text_margin, traffic_y + traffic_line_spacing * 5), 
                           cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.5, (255, 120, 120), max(text_thickness - 2, 1), cv2.LINE_AA)
            elif animal_detected_start is not None:
                cv2.putText(frame, 'ANIMAL SAFETY MODE ACTIVE', (traffic_x + traffic_text_margin, traffic_y + traffic_line_spacing * 5), 
                           cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.5, (255, 255, 120), max(text_thickness - 2, 1), cv2.LINE_AA)
            else:
                cv2.putText(frame, 'AUTOMATED TRAFFIC MANAGEMENT', (traffic_x + traffic_text_margin, traffic_y + traffic_line_spacing * 5), 
                           cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.5, traffic_text_color, max(text_thickness - 2, 1), cv2.LINE_AA)
    
    def get_statistics(self):
        """Get comprehensive analysis statistics including traffic control, violence and animal detection"""
        elapsed = time.time() - self.stats['analysis_start']
        fps = self.stats['total_frames'] / elapsed if elapsed > 0 else 0
        
        base_stats = {
            'total_frames': self.stats['total_frames'],
            'vehicles_detected': self.stats['vehicles_detected'],
            'elapsed_time': elapsed,
            'fps': fps,
            'vehicles_per_frame': self.stats['vehicles_detected'] / max(1, self.stats['total_frames'])
        }
        
        # Add traffic control statistics
        base_stats['traffic_control'] = {
            'enabled': self.enable_traffic_control,
            'sumo_connected': self.sumo_enabled,
            'control_actions': self.stats.get('traffic_control_actions', 0),
            'left_lane_actions': self.stats.get('left_lane_control_actions', 0),
            'right_lane_actions': self.stats.get('right_lane_control_actions', 0),
            'emergency_stops': self.stats.get('emergency_stops', 0),
            'sms_alerts_enabled': TWILIO_AVAILABLE,
            'left_lane_threshold': LEFT_LANE_THRESHOLD,
            'right_lane_threshold': RIGHT_LANE_THRESHOLD,
            'density_imbalance_ratio': DENSITY_IMBALANCE_RATIO,
            'violence_alert_duration': VIOLENCE_ALERT_DURATION,
            'animal_alert_duration': ANIMAL_ALERT_DURATION
        }
        
        # Add violence detection statistics if available
        if self.violence_detector:
            violence_stats = self.violence_detector.get_statistics()
            base_stats['violence_detection'] = violence_stats
            base_stats['unacknowledged_alerts'] = self.alert_manager.get_unacknowledged_count()
            base_stats['violence_alerts_sent'] = self.stats.get('violence_alerts_sent', 0)
        
        # Add animal detection statistics
        recent_animal_alerts = self.animal_alert_manager.get_recent_alerts(hours=24)
        base_stats['animal_detection'] = {
            'recent_alerts_24h': len(recent_animal_alerts),
            'current_alert_active': self.current_animal_alert is not None,
            'detection_interval': self.animal_detection_interval,
            'traffic_adjustments': self.stats.get('animal_traffic_adjustments', 0)
        }
        
        return base_stats
    
    def cleanup(self):
        """Cleanup resources including traffic control, violence and animal detection"""
        # Stop traffic control system
        if self.enable_traffic_control and self.sumo_enabled:
            stop_sumo()
            print("[TRAFFIC CONTROL] SUMO simulation stopped")
        
        # Stop violence detection
        if self.violence_detector:
            self.violence_detector.stop_processing()
            print("[SHIELD] Violence detection stopped")
            
        print("[ANIMAL] Animal detection cleaned up")
        print("[SYSTEM] All resources cleaned up successfully")
    
    def _process_violence_results(self):
        """Process pending violence detection results"""
        if not self.violence_detector:
            return
        
        # Check for new detection results
        detection = self.violence_detector.get_latest_detection()
        if detection:
            # Create alert
            self.alert_manager.process_detection(detection)
            
            # Set current alert for display
            self.current_violence_alert = detection
            self.alert_display_time = time.time()
            
            print(f"[ALERT] VIOLENCE DETECTED: {detection['max_confidence']:.2f} confidence")
        
        # Clear alert after duration
        if (self.current_violence_alert and 
            time.time() - self.alert_display_time > self.alert_duration):
            self.current_violence_alert = None

    def _process_animal_detection(self, frame):
        """Process animal detection for current frame"""
        try:
            # Detect animals in frame
            animal_counts, detections = self.animal_detector.detect_animals(frame)
            
            # Check if any animals detected
            total_animals = sum(animal_counts.values())
            if total_animals > 0:
                print(f"[ANIMAL] Frame {self.stats['total_frames']}: {total_animals} animal(s) detected")
                # Debug: show detailed counts
                for animal_type, count in animal_counts.items():
                    if count > 0:
                        print(f"  - {animal_type.upper()}: {count}")
                
                # Debug: show detection details
                for detection in detections:
                    print(f"    {detection['type']}: confidence={detection['confidence']:.3f}")
                
                # Check if we should alert for any detected animal
                should_alert = False
                for animal_type, count in animal_counts.items():
                    if count > 0 and self.animal_alert_manager.should_alert(animal_type):
                        should_alert = True
                        break
                
                if should_alert:
                    # Save evidence and create alert
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    evidence_path = loop.run_until_complete(
                        self.animal_alert_manager.save_evidence(
                            frame, animal_counts, detections, self.stats['total_frames']
                        )
                    )
                    
                    if evidence_path:
                        alert = loop.run_until_complete(
                            self.animal_alert_manager.create_alert(
                                animal_counts, detections, self.stats['total_frames'], evidence_path
                            )
                        )
                        
                        # Set current alert for display
                        self.current_animal_alert = {
                            'animal_counts': animal_counts,
                            'detections': detections,
                            'total_animals': total_animals,
                            'severity': alert['severity'],
                            'alert': alert
                        }
                        self.animal_alert_display_time = time.time()
                        
                        print(f"[ANIMAL ALERT] {total_animals} animal(s) detected - Severity: {alert['severity']}")
                        for animal, count in animal_counts.items():
                            if count > 0:
                                print(f"  - {animal.upper()}: {count}")
                    
                    loop.close()
            else:
                # Debug: show when no animals detected (occasionally)
                if self.stats['total_frames'] % 150 == 0:  # Every 150 frames
                    print(f"[ANIMAL] Frame {self.stats['total_frames']}: No animals detected")
                
        except Exception as e:
            print(f"❌ Error in animal detection: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_animal_results(self):
        """Process pending animal detection alert display"""
        # Clear alert after duration
        if (self.current_animal_alert and 
            time.time() - self.animal_alert_display_time > self.animal_alert_duration):
            self.current_animal_alert = None
    
    def _create_animal_alert_overlay(self, frame, animal_alert):
        """Create enhanced animal detection alert overlay with resolution-adaptive crisp rendering"""
        try:
            # Annotate frame with animal detections
            if 'detections' in animal_alert:
                frame = self.animal_detector.annotate_frame(frame, animal_alert['detections'])
            
            # Resolution-adaptive banner sizing
            frame_height, frame_width = frame.shape[:2]
            pixel_density = (frame_width * frame_height) / (1920 * 1080)
            density_factor = min(max(pixel_density, 0.3), 2.5)
            
            banner_height = max(int(110 * density_factor), 90)
            
            # Multi-tier font scaling
            if frame_width >= 3840:  # 4K
                base_font_scale = 2.0
                thickness_multiplier = 4
            elif frame_width >= 2560:  # 2K
                base_font_scale = 1.6
                thickness_multiplier = 3
            elif frame_width >= 1920:  # 1080p
                base_font_scale = 1.2
                thickness_multiplier = 3
            elif frame_width >= 1280:  # 720p
                base_font_scale = 0.9
                thickness_multiplier = 2
            else:  # Lower resolutions
                base_font_scale = 0.7
                thickness_multiplier = 2
            
            font_scale = base_font_scale * density_factor
            font_scale = max(0.6, min(font_scale, 2.5))
            text_thickness = max(int(thickness_multiplier * density_factor), 2)
            border_thickness = max(int(4 * density_factor), 3)
            
            # Create dark background for better contrast
            cv2.rectangle(frame, (0, 0), (frame_width, banner_height), (0, 0, 0), -1)
            
            # Create colored alert banner with crisp edges
            cv2.rectangle(frame, (5, 5), (frame_width - 5, banner_height - 5), (0, 220, 220), -1)
            
            # Anti-aliased white border for definition
            cv2.rectangle(frame, (5, 5), (frame_width - 5, banner_height - 5), (255, 255, 255), border_thickness)
            
            # Enhanced alert text with precise positioning and anti-aliasing
            alert_text = f"ANIMAL DETECTION ALERT - {animal_alert['total_animals']} ANIMALS DETECTED"
            
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, text_thickness)[0]
            text_x = (frame_width - text_size[0]) // 2
            
            # Main alert text with shadow effect and anti-aliasing
            shadow_offset = max(int(3 * density_factor), 2)
            cv2.putText(frame, alert_text, (text_x + shadow_offset, 35 + shadow_offset), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)
            cv2.putText(frame, alert_text, (text_x, 35), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
            
            # Animal details with crisp rendering and better formatting
            y_offset = max(int(70 * density_factor), 60)
            x_offset = max(int(25 * density_factor), 20)
            detail_spacing = max(int(50 * density_factor), 40)
            
            for animal, count in animal_alert['animal_counts'].items():
                if count > 0:
                    detail_text = f"{animal.upper()}: {count} DETECTED"
                    # Background for text readability
                    text_size_detail = cv2.getTextSize(detail_text, cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.7, text_thickness - 1)[0]
                    
                    # Dark background for text
                    cv2.rectangle(frame, (x_offset - 8, y_offset - 22), 
                                 (x_offset + text_size_detail[0] + 8, y_offset + 8), (0, 0, 0), -1)
                    
                    # Text with anti-aliasing
                    cv2.putText(frame, detail_text, (x_offset, y_offset), 
                               cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.7, (255, 255, 255), text_thickness - 1, cv2.LINE_AA)
                    x_offset += text_size_detail[0] + detail_spacing
            
            # Enhanced severity indicator with crisp rendering
            severity = animal_alert.get('severity', 'LOW')
            severity_colors = {
                'LOW': (0, 255, 0),      # Green
                'MEDIUM': (0, 165, 255), # Orange  
                'HIGH': (0, 0, 255)      # Red
            }
            severity_color = severity_colors.get(severity, (255, 255, 255))
            
            severity_text = f"PRIORITY LEVEL: {severity}"
            severity_text_size = cv2.getTextSize(severity_text, cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.8, text_thickness)[0]
            severity_x = frame_width - severity_text_size[0] - max(int(25 * density_factor), 20)
            
            # Background for severity indicator
            cv2.rectangle(frame, (severity_x - 12, 8), 
                         (severity_x + severity_text_size[0] + 12, 45), (0, 0, 0), -1)
            
            # Severity text with anti-aliasing
            cv2.putText(frame, severity_text, (severity_x, 35), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale * 0.8, severity_color, text_thickness, cv2.LINE_AA)
            
            return frame
            
        except Exception as e:
            print(f"Error creating animal alert overlay: {e}")
            return frame

def initialize_video_source(source):
    """Initialize video source with proper handling for different types"""
    print(f"\n🔌 Initializing video source: {source}")
    
    # Handle YouTube URLs
    if isinstance(source, str) and ('youtube.com' in source or 'youtu.be' in source):
        print("🎥 YouTube stream detected")
        return YouTubeStreamManager(source)
    
    # Handle webcam sources
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        camera_id = int(source)
        print(f"📷 Webcam detected (ID: {camera_id})")
        return initialize_webcam(camera_id)
    
    # Handle other sources (files, IP cameras, RTSP)
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

def initialize_webcam(camera_id):
    """Enhanced webcam initialization with better error handling and camera detection"""
    print(f"📷 Initializing webcam (ID: {camera_id})...")
    
    # First, check available cameras
    available_cameras = detect_available_cameras()
    if not available_cameras:
        print("❌ No cameras detected on this system")
        return None
    
    print(f"📋 Available cameras: {available_cameras}")
    
    # If requested camera is not available, try alternatives
    if camera_id not in available_cameras:
        print(f"⚠️  Camera {camera_id} not available, trying alternatives...")
        camera_id = available_cameras[0]
        print(f"🔄 Switching to camera {camera_id}")
    
    # Try different backends specifically for webcams
    webcam_backends = [
        (cv2.CAP_DSHOW, "DirectShow (Windows)"),
        (cv2.CAP_MSMF, "Media Foundation (Windows)"),
        (cv2.CAP_ANY, "Auto-detect"),
        (cv2.CAP_V4L2, "Video4Linux2 (Linux)"),
        (cv2.CAP_GSTREAMER, "GStreamer")
    ]
    
    for backend, backend_name in webcam_backends:
        try:
            print(f"🔄 Trying {backend_name}...")
            cap = cv2.VideoCapture(camera_id, backend)
            
            if cap.isOpened():
                # Configure webcam settings for better performance
                try:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    print(f"   📐 Configured resolution: 1280x720 @ 30fps")
                except Exception as e:
                    print(f"   ⚠️ Could not set webcam properties: {e}")
                
                # Test frame reading with retry
                for attempt in range(3):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"✅ Webcam connected using {backend_name}")
                        print(f"📐 Actual resolution: {width}x{height}")
                        
                        # Get actual FPS
                        try:
                            actual_fps = cap.get(cv2.CAP_PROP_FPS)
                            if actual_fps > 0:
                                print(f"🎬 Camera FPS: {actual_fps}")
                        except:
                            pass
                        
                        return cap
                    else:
                        print(f"   ❌ Frame read attempt {attempt + 1} failed")
                        time.sleep(0.5)
                
                cap.release()
                print(f"   ❌ {backend_name} opened but could not read frames")
            else:
                print(f"   ❌ {backend_name} failed to open camera")
                
        except Exception as e:
            print(f"   ❌ {backend_name} error: {e}")
    
    print("❌ All webcam backends failed")
    print("\n🔧 Troubleshooting tips:")
    print("   • Check if camera is being used by another application")
    print("   • Try closing other video applications (Zoom, Teams, etc.)")
    print("   • Check Windows Camera privacy settings")
    print("   • Try a different camera ID (0, 1, 2...)")
    print("   • Consider using a video file instead: --source videos/your_video.mp4")
    
    return None

def detect_available_cameras(max_cameras=10):
    """Detect available camera indices"""
    print("🔍 Detecting available cameras...")
    available_cameras = []
    
    for i in range(max_cameras):
        try:
            # Quick test with minimal timeout
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow for faster detection on Windows
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    available_cameras.append(i)
                    # Get camera info if possible
                    try:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"   📷 Camera {i}: {width}x{height}")
                    except:
                        print(f"   📷 Camera {i}: Available")
            cap.release()
        except:
            pass
    
    return available_cameras

def get_source_selection():
    """Interactive source selection with improved webcam handling"""
    print("\n=== INPUT SOURCE SELECTION ===")
    print("1. Video file")
    print("2. Webcam (detect available cameras)")
    print("3. IP Camera/HTTP Stream")
    print("4. RTSP Stream")
    print("5. YouTube Live Stream")
    
    choice = input("Select input source (1-5): ").strip()
    
    if choice == "1":
        path = input("Enter video file path (or press Enter for default): ").strip().strip('"')
        return path if path else "videos/indian traffic.mp4"
    elif choice == "2":
        # Detect available cameras first
        print("\n🔍 Scanning for available cameras...")
        available_cameras = detect_available_cameras()
        
        if not available_cameras:
            print("❌ No cameras detected!")
            print("🔧 Troubleshooting:")
            print("   • Ensure no other applications are using the camera")
            print("   • Check Windows Camera privacy settings")
            print("   • Try connecting an external USB camera")
            fallback = input("\nWould you like to try a video file instead? (y/N): ").strip().lower()
            if fallback == 'y':
                path = input("Enter video file path: ").strip().strip('"')
                return path if path else "videos/indian traffic.mp4"
            else:
                print("Using default camera ID 0 (may fail)")
                return 0
        else:
            print(f"📷 Found {len(available_cameras)} camera(s): {available_cameras}")
            if len(available_cameras) == 1:
                selected_cam = available_cameras[0]
                print(f"Using camera {selected_cam}")
                return selected_cam
            else:
                cam_input = input(f"Select camera ID {available_cameras} (or press Enter for {available_cameras[0]}): ").strip()
                if cam_input.isdigit() and int(cam_input) in available_cameras:
                    return int(cam_input)
                else:
                    return available_cameras[0]
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
        return "videos/indian traffic.mp4"

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

def reset_video_source(cap, source):
    """Reset video source to beginning after calibration"""
    print("🔄 Resetting video to beginning for traffic analysis...")
    
    if isinstance(cap, YouTubeStreamManager):
        # For YouTube streams, force a fresh connection to get latest stream
        print("   YouTube stream - getting fresh connection...")
        cap.current_stream_url = None  # Force URL refresh
        cap.last_refresh = 0  # Force refresh
        if cap.connect():
            print("   ✅ YouTube stream refreshed successfully")
        else:
            print("   ⚠️ YouTube stream refresh failed, continuing with current connection")
        return cap
    else:
        # For file sources, set position to beginning
        if isinstance(source, str) and not source.startswith(('http://', 'https://', 'rtsp://')):
            # It's a file, reset to beginning
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Verify reset worked by checking position
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if current_pos == 0:
                    print("   ✅ Video file reset to beginning")
                else:
                    print(f"   ⚠️ Video reset attempted, position: {current_pos}")
            except Exception as e:
                print(f"   ⚠️ Could not reset video position: {e}")
        else:
            # It's a live stream/camera, can't reset
            print("   Live stream - continuing from current position")
        return cap

def main():
    """Main function with integrated traffic control system"""
    parser = argparse.ArgumentParser(description='IoT-Based Traffic Regulation System with Real-time Detection and Control')
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
    
    # Violence detection arguments
    parser.add_argument('--violence-api-user', type=str, 
                       help='Sightengine API user ID for violence detection')
    parser.add_argument('--violence-api-secret', type=str, 
                       help='Sightengine API secret for violence detection')
    parser.add_argument('--disable-violence', action='store_true', 
                       help='Disable violence detection even if API credentials are available')
    parser.add_argument('--violence-threshold', type=float, default=0.7, 
                       help='Violence detection threshold (0.0-1.0)')
    parser.add_argument('--violence-interval', type=int, default=30, 
                       help='Check every N frames for violence detection')
    
    # Traffic control arguments
    parser.add_argument('--disable-traffic-control', action='store_true',
                       help='Disable automated traffic control system')
    parser.add_argument('--left-lane-threshold', type=int, default=8,
                       help='Vehicle count threshold for left lane control activation')
    parser.add_argument('--right-lane-threshold', type=int, default=8,
                       help='Vehicle count threshold for right lane control activation')
    parser.add_argument('--density-imbalance-ratio', type=float, default=2.0,
                       help='Lane density ratio to trigger priority switching')
    parser.add_argument('--violence-alert-duration', type=int, default=100,
                       help='Seconds of continuous violence before SMS alert')
    parser.add_argument('--animal-alert-duration', type=int, default=20,
                       help='Seconds of animal presence before traffic adjustment')
    parser.add_argument('--emergency-number', type=str, default="+919877035742",
                       help='Phone number for emergency SMS alerts')
    
    args = parser.parse_args()
    
    # Configure violence detection from command line arguments
    if args.violence_api_user:
        config.SIGHTENGINE_API_USER = args.violence_api_user
    if args.violence_api_secret:
        config.SIGHTENGINE_API_SECRET = args.violence_api_secret
    if args.disable_violence:
        config.VIOLENCE_DETECTION_ENABLED = False
    if args.violence_threshold:
        config.VIOLENCE_THRESHOLD = args.violence_threshold
    if args.violence_interval:
        config.VIOLENCE_CHECK_INTERVAL = args.violence_interval
    
    # Reload API credentials after potential updates
    config.load_api_credentials()
    
    # Configure traffic control from command line arguments
    global LEFT_LANE_THRESHOLD, RIGHT_LANE_THRESHOLD, DENSITY_IMBALANCE_RATIO, VIOLENCE_ALERT_DURATION, ANIMAL_ALERT_DURATION, POLICE_NUMBER
    if args.left_lane_threshold:
        LEFT_LANE_THRESHOLD = args.left_lane_threshold
    if args.right_lane_threshold:
        RIGHT_LANE_THRESHOLD = args.right_lane_threshold
    if args.density_imbalance_ratio:
        DENSITY_IMBALANCE_RATIO = args.density_imbalance_ratio
    if args.violence_alert_duration:
        VIOLENCE_ALERT_DURATION = args.violence_alert_duration
    if args.animal_alert_duration:
        ANIMAL_ALERT_DURATION = args.animal_alert_duration
    if args.emergency_number:
        POLICE_NUMBER = args.emergency_number
    
    enable_traffic_control = not args.disable_traffic_control
    
    print("🚗 IoT-Based Traffic Regulation System")
    print("🚦 [TRAFFIC CONTROL] With Real-time Automated Traffic Management")
    print("[SHIELD] With Advanced Violence Detection")
    print("[ANIMAL] With Animal Detection & Safety")
    print("=" * 70)
    
    # Show traffic control status
    if enable_traffic_control:
        print("✅ Traffic Control: ENABLED")
        print(f"   SUMO Available: {SUMO_AVAILABLE}")
        print(f"   SUMO Config Path: {SUMO_CONFIG_PATH}")
        print(f"   Left Lane Threshold: {LEFT_LANE_THRESHOLD} vehicles")
        print(f"   Right Lane Threshold: {RIGHT_LANE_THRESHOLD} vehicles")
        print(f"   Density Imbalance Ratio: {DENSITY_IMBALANCE_RATIO}")
        print(f"   Violence Alert Duration: {VIOLENCE_ALERT_DURATION}s")
        print(f"   Animal Alert Duration: {ANIMAL_ALERT_DURATION}s")
        print(f"   Emergency Number: {POLICE_NUMBER}")
        print(f"   SMS Alerts: {'ENABLED' if TWILIO_AVAILABLE else 'DISABLED'}")
    else:
        print("⚠️ Traffic Control: DISABLED")
    
    # Show violence detection status
    if config.VIOLENCE_DETECTION_ENABLED:
        print("✅ Violence detection: ENABLED")
        print(f"   Threshold: {config.VIOLENCE_THRESHOLD}")
        print(f"   Check interval: {config.VIOLENCE_CHECK_INTERVAL} frames")
        print(f"   Models: {', '.join(config.VIOLENCE_MODELS)}")
    else:
        print("[WARNING] Violence detection: DISABLED")
        if not config.SIGHTENGINE_API_USER or not config.SIGHTENGINE_API_SECRET:
            print("   Reason: Missing API credentials")
        else:
            print("   Reason: Explicitly disabled")
    print("=" * 70)
    
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
    
    # Reset video to beginning after sample frame read (for file sources)
    if not isinstance(cap, YouTubeStreamManager) and isinstance(source, str) and not source.startswith(('http://', 'https://', 'rtsp://')):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("🔄 Video reset to beginning after sample frame read")
    
    # Configure lanes
    if args.no_calibration:
        print("\n🎯 Using saved/default lane configuration")
        lane_config = load_saved_config()
    else:
        if args.headless:
            print("\n🎯 Headless mode: using saved/default configuration")
            lane_config = load_saved_config()
        else:
            print("\n🎯 Starting interactive calibration...")
            # For YouTube streams, use the existing stream manager
            if isinstance(cap, YouTubeStreamManager):
                print("   Using existing YouTube stream for calibration...")
                calibration_cap = cap
            else:
                calibration_cap = cap
            
            calibrator = StreamCalibrator(calibration_cap)
            lane_config = calibrator.calibrate_stream()
            
            # No need to clean up for YouTube since we used the same stream
            
            if lane_config is None:
                print("Calibration cancelled")
                if hasattr(cap, 'release'):
                    cap.release()
                return
            
            # Reset video source to beginning after calibration
            cap = reset_video_source(cap, source)
    
    print(f"\n✅ Configuration loaded:")
    print(f"   Polygons: {len(lane_config['polygons'])}")
    print(f"   Lane threshold: {lane_config['lane_threshold']}")
    print(f"   Detection area: {lane_config['detection_area']}")
    
    # Initialize traffic analyzer with traffic control integration
    analyzer = TrafficAnalyzer(
        model, 
        lane_config, 
        headless=args.headless,
        enable_traffic_control=enable_traffic_control
    )
    
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
    
    print(f"\n[ROCKET] Starting traffic analysis...")
    print(f"Mode: {'Headless' if args.headless else 'GUI'}")
    if not args.headless:
        if isinstance(cap, YouTubeStreamManager):
            print("Controls: 'q' to quit, 's' to save frame, 'p' to pause, 'r' to refresh stream")
        else:
            print("Controls: 'q' to quit, 's' to save frame, 'p' to pause")
    
    # Main processing loop
    frame_count = 0
    fps_counter = 0
    fps_start = time.time()
    paused = False
    
    try:
        while True:
            # Read frame with improved error handling
            if isinstance(cap, YouTubeStreamManager):
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ YouTube stream interrupted, attempting recovery...")
                    time.sleep(2)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ End of video or read failed")
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
                        
                        # Add stream status for YouTube
                        if isinstance(cap, YouTubeStreamManager):
                            status_text = f"Stream: {'Stable' if cap.consecutive_failures == 0 else f'{cap.consecutive_failures} failures'}"
                            cv2.putText(processed_frame, status_text, 
                                       (10, processed_frame.shape[0] - 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
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
                    stream_status = ""
                    if isinstance(cap, YouTubeStreamManager):
                        stream_status = f", Stream: {cap.consecutive_failures} failures"
                    print(f"Frame {frame_count}: L={vehicles_left}, R={vehicles_right}, "
                          f"FPS={stats['fps']:.1f}, Total vehicles={stats['vehicles_detected']}{stream_status}")
            
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
                elif key == ord('r') and isinstance(cap, YouTubeStreamManager):
                    print("🔄 Manual stream refresh requested...")
                    cap.current_stream_url = None
                    cap.last_refresh = 0
                    cap.consecutive_failures = 0
            
    except KeyboardInterrupt:
        print("\n⏹️  Analysis stopped by user")
    
    finally:
        # Cleanup
        analyzer.cleanup()  # This will stop violence detection
        
        if hasattr(cap, 'release'):
            cap.release()
        if out_writer:
            out_writer.release()
        if not args.headless:
            cv2.destroyAllWindows()
        
        # Print final statistics
        stats = analyzer.get_statistics()
        print(f"\n📊 Final IoT Traffic Regulation System Statistics:")
        print(f"   Total frames processed: {stats['total_frames']}")
        print(f"   Total vehicles detected: {stats['vehicles_detected']}")
        print(f"   Average FPS: {stats['fps']:.2f}")
        print(f"   Vehicles per frame: {stats['vehicles_per_frame']:.2f}")
        print(f"   Analysis duration: {stats['elapsed_time']:.2f} seconds")
        
        # Print traffic control statistics
        if 'traffic_control' in stats:
            tc_stats = stats['traffic_control']
            print(f"\n🚦 [TRAFFIC CONTROL] Lane-Specific System Statistics:")
            print(f"   Traffic control enabled: {tc_stats['enabled']}")
            print(f"   SUMO simulation connected: {tc_stats['sumo_connected']}")
            print(f"   Total control actions: {tc_stats['control_actions']}")
            print(f"   Left lane priority actions: {tc_stats['left_lane_actions']}")
            print(f"   Right lane priority actions: {tc_stats['right_lane_actions']}")
            print(f"   Emergency stops triggered: {tc_stats['emergency_stops']}")
            print(f"   Left lane threshold: {tc_stats['left_lane_threshold']} vehicles")
            print(f"   Right lane threshold: {tc_stats['right_lane_threshold']} vehicles")
            print(f"   Density imbalance ratio: {tc_stats['density_imbalance_ratio']}")
            print(f"   SMS alerts enabled: {tc_stats['sms_alerts_enabled']}")
            print(f"   Violence alert duration: {tc_stats['violence_alert_duration']}s")
            print(f"   Animal alert duration: {tc_stats['animal_alert_duration']}s")
        
        # Print violence detection statistics if available
        if 'violence_detection' in stats and stats['violence_detection']['enabled']:
            vio_stats = stats['violence_detection']
            print(f"\n🛡️ [SHIELD] Violence Detection Statistics:")
            print(f"   Total checks: {vio_stats['total_checks']}")
            print(f"   Violence detected: {vio_stats['violence_detected']}")
            print(f"   Detection rate: {vio_stats['detection_rate']:.2%}")
            print(f"   API errors: {vio_stats['api_errors']}")
            print(f"   Avg processing time: {vio_stats['avg_processing_time']:.3f}s")
            if stats['unacknowledged_alerts'] > 0:
                print(f"   ⚠️  Unacknowledged alerts: {stats['unacknowledged_alerts']}")
            if stats.get('violence_alerts_sent', 0) > 0:
                print(f"   📲 Emergency SMS alerts sent: {stats['violence_alerts_sent']}")
        
        # Print animal detection statistics
        if 'animal_detection' in stats:
            animal_stats = stats['animal_detection']
            print(f"\n🐕 [ANIMAL] Detection Statistics:")
            print(f"   Recent alerts (24h): {animal_stats['recent_alerts_24h']}")
            print(f"   Current alert active: {animal_stats['current_alert_active']}")
            print(f"   Detection interval: every {animal_stats['detection_interval']} frames")
            if animal_stats.get('traffic_adjustments', 0) > 0:
                print(f"   🚦 Traffic adjustments made: {animal_stats['traffic_adjustments']}")
        
        print(f"\n✅ IoT Traffic Regulation System completed successfully!")
        print(f"   All systems: Vehicle Detection ✓ | Animal Detection ✓ | Violence Detection ✓ | Traffic Control ✓")

if __name__ == "__main__":
    main()
