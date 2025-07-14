#!/usr/bin/env python3
"""
⚙️ Configuration and Settings for Traffic Analysis System
Centralized configuration management for all system components

Author: Pratham Handa
GitHub: https://github.com/prathamhanda/IoT-Based_Traffic_Regulation
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import sys

class Config:
    """Main configuration class for traffic analysis system"""
    
    def __init__(self):
        # Default model settings
        self.MODEL_PATH = "models/best.pt"
        self.MODEL_CONFIDENCE = 0.4
        self.MODEL_IOU_THRESHOLD = 0.45
        self.MODEL_IMAGE_SIZE = 640
        
        # Video processing settings
        self.VIDEO_BACKENDS = ['cv2.CAP_FFMPEG', 'cv2.CAP_GSTREAMER', 'cv2.CAP_ANY']
        self.BUFFER_SIZE = 1
        self.DEFAULT_FPS = 30
        self.FRAME_SKIP = 0  # Skip frames for performance
        
        # Stream settings
        self.STREAM_TIMEOUT_MS = 10000
        self.READ_TIMEOUT_MS = 5000
        self.RECONNECT_ATTEMPTS = 5
        self.RECONNECT_DELAY = 2.0
        self.YOUTUBE_REFRESH_INTERVAL = 300  # 5 minutes
        
        # Display settings
        self.DISPLAY_WIDTH = 1200
        self.DISPLAY_HEIGHT = 675
        self.MAX_DISPLAY_WIDTH = 1024
        self.MAX_DISPLAY_HEIGHT = 576
        self.FONT_SCALE_FACTOR = 1.5
        self.LINE_THICKNESS = 2
        
        # Detection settings
        self.HEAVY_TRAFFIC_THRESHOLD = 8
        self.VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.LANE_COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
        # Violence Detection API settings
        self.VIOLENCE_DETECTION_ENABLED = False
        self.SIGHTENGINE_API_USER = ""  # Set via environment or config file
        self.SIGHTENGINE_API_SECRET = ""  # Set via environment or config file
        self.VIOLENCE_CHECK_INTERVAL = 30  # Check every 30 frames
        self.VIOLENCE_MODELS = ["violence", "gore", "weapon"]  # Models to apply
        self.VIOLENCE_THRESHOLD = 0.7  # Threshold for violence detection
        self.VIOLENCE_ALERT_ENABLED = True
        self.VIOLENCE_LOG_ENABLED = True
        self.VIOLENCE_SAVE_EVIDENCE = True  # Save frames with violence detection
        
        # Output settings
        self.OUTPUT_CODEC = 'XVID'
        self.OUTPUT_FPS = 20
        self.SAVE_FRAME_QUALITY = 95
        
        # System settings
        self.MAX_FRAMES_DEFAULT = None
        self.STATISTICS_INTERVAL = 30  # frames
        self.MEMORY_CLEANUP_INTERVAL = 100  # frames
        
        # File paths
        self.CONFIG_FILE = "lane_config.json"
        self.LOGS_DIR = "logs"
        self.OUTPUT_DIR = "output"
        self.CACHE_DIR = "cache"
        
        # YouTube settings
        self.YOUTUBE_FORMAT = 'best[height<=720][fps<=30]'
        self.YOUTUBE_QUALITY_FALLBACKS = [
            'best[height<=720]',
            'best[height<=480]',
            'best[height<=360]',
            'best'
        ]
        
        # Default lane configuration
        self.DEFAULT_LANE_CONFIG = {
            'polygons': [
                [(465, 350), (609, 350), (510, 630), (2, 630)],
                [(678, 350), (815, 350), (1203, 630), (743, 630)]
            ],
            'lane_threshold': 609,
            'detection_area': (325, 635)
        }
        
        # Load user configuration if exists
        self.load_user_config()
        
        # Load API credentials from environment
        self.load_api_credentials()
    
    def load_user_config(self, config_file: str = "config.json"):
        """Load user configuration from file"""
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                
                # Update configuration with user values
                for key, value in user_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        
                print(f"✅ User configuration loaded from {config_file}")
            except Exception as e:
                print(f"⚠️  Could not load user config: {e}")
    
    def load_api_credentials(self):
        """Load API credentials from environment variables or config"""
        # Try to load from environment variables first
        self.SIGHTENGINE_API_USER = os.getenv('SIGHTENGINE_API_USER', self.SIGHTENGINE_API_USER)
        self.SIGHTENGINE_API_SECRET = os.getenv('SIGHTENGINE_API_SECRET', self.SIGHTENGINE_API_SECRET)
        
        # Enable violence detection only if credentials are available
        if self.SIGHTENGINE_API_USER and self.SIGHTENGINE_API_SECRET:
            self.VIOLENCE_DETECTION_ENABLED = True
            print("✅ Violence detection API credentials loaded")
        else:
            self.VIOLENCE_DETECTION_ENABLED = False
            print("⚠️  Violence detection disabled - API credentials not found")
    
    def save_user_config(self, config_file: str = "config.json"):
        """Save current configuration to file"""
        try:
            config_dict = {
                'MODEL_PATH': self.MODEL_PATH,
                'MODEL_CONFIDENCE': self.MODEL_CONFIDENCE,
                'MODEL_IOU_THRESHOLD': self.MODEL_IOU_THRESHOLD,
                'STREAM_TIMEOUT_MS': self.STREAM_TIMEOUT_MS,
                'READ_TIMEOUT_MS': self.READ_TIMEOUT_MS,
                'HEAVY_TRAFFIC_THRESHOLD': self.HEAVY_TRAFFIC_THRESHOLD,
                'DISPLAY_WIDTH': self.DISPLAY_WIDTH,
                'DISPLAY_HEIGHT': self.DISPLAY_HEIGHT,
                'YOUTUBE_REFRESH_INTERVAL': self.YOUTUBE_REFRESH_INTERVAL,
                'VIOLENCE_CHECK_INTERVAL': self.VIOLENCE_CHECK_INTERVAL,
                'VIOLENCE_MODELS': self.VIOLENCE_MODELS,
                'VIOLENCE_THRESHOLD': self.VIOLENCE_THRESHOLD,
                'VIOLENCE_ALERT_ENABLED': self.VIOLENCE_ALERT_ENABLED,
                'VIOLENCE_LOG_ENABLED': self.VIOLENCE_LOG_ENABLED,
                'VIOLENCE_SAVE_EVIDENCE': self.VIOLENCE_SAVE_EVIDENCE
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
                
            print(f"✅ Configuration saved to {config_file}")
        except Exception as e:
            print(f"❌ Could not save config: {e}")
    
    def get_lane_config(self) -> Dict:
        """Get lane configuration (saved or default)"""
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                return {
                    'polygons': config['polygons'],
                    'lane_threshold': config['lane_threshold'],
                    'detection_area': config['detection_area']
                }
            except Exception as e:
                print(f"⚠️  Could not load lane config: {e}")
        
        return self.DEFAULT_LANE_CONFIG.copy()
    
    def save_lane_config(self, config: Dict):
        """Save lane configuration"""
        try:
            config_data = {
                'timestamp': datetime.now().isoformat(),
                'polygons': config['polygons'],
                'lane_threshold': config['lane_threshold'],
                'detection_area': config['detection_area']
            }
            
            with open(self.CONFIG_FILE, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            print(f"✅ Lane configuration saved to {self.CONFIG_FILE}")
        except Exception as e:
            print(f"❌ Could not save lane config: {e}")
    
    def ensure_directories(self):
        """Create necessary directories"""
        directories = [self.LOGS_DIR, self.OUTPUT_DIR, self.CACHE_DIR]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Created directory: {directory}")
    
    def get_youtube_opts(self, quality: str = None) -> Dict:
        """Get yt-dlp options for YouTube streams"""
        format_selector = quality or self.YOUTUBE_FORMAT
        
        return {
            'format': format_selector,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'socket_timeout': 30,
            'retries': 3
        }
    
    def get_opencv_backends(self) -> List[int]:
        """Get OpenCV backend constants"""
        import cv2
        backends = []
        
        backend_map = {
            'cv2.CAP_FFMPEG': cv2.CAP_FFMPEG,
            'cv2.CAP_GSTREAMER': cv2.CAP_GSTREAMER if hasattr(cv2, 'CAP_GSTREAMER') else None,
            'cv2.CAP_ANY': cv2.CAP_ANY
        }
        
        for backend_name in self.VIDEO_BACKENDS:
            backend = backend_map.get(backend_name)
            if backend is not None:
                backends.append(backend)
        
        return backends
    
    def validate_model_path(self) -> bool:
        """Validate model file exists"""
        if not os.path.exists(self.MODEL_PATH):
            print(f"❌ Model file not found: {self.MODEL_PATH}")
            return False
        
        file_size = os.path.getsize(self.MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model found: {self.MODEL_PATH} ({file_size:.1f} MB)")
        return True
    
    def get_display_scale(self, frame_width: int, frame_height: int) -> float:
        """Calculate display scale for frame"""
        if frame_width <= self.MAX_DISPLAY_WIDTH and frame_height <= self.MAX_DISPLAY_HEIGHT:
            return 1.0
        
        width_scale = self.MAX_DISPLAY_WIDTH / frame_width
        height_scale = self.MAX_DISPLAY_HEIGHT / frame_height
        return min(width_scale, height_scale)
    
    def get_font_scale(self, frame_width: int, frame_height: int) -> float:
        """Calculate font scale based on frame size"""
        base_scale = min(frame_width / 1920, frame_height / 1080) * self.FONT_SCALE_FACTOR
        return max(0.5, min(base_scale, 2.0))
    
    def print_config(self):
        """Print current configuration"""
        print("\n⚙️ Current Configuration")
        print("=" * 50)
        
        # Model settings
        print(f"🎯 Model: {self.MODEL_PATH}")
        print(f"   Confidence: {self.MODEL_CONFIDENCE}")
        print(f"   Image size: {self.MODEL_IMAGE_SIZE}")
        
        # Stream settings
        print(f"📹 Stream timeout: {self.STREAM_TIMEOUT_MS}ms")
        print(f"   Read timeout: {self.READ_TIMEOUT_MS}ms")
        print(f"   Reconnect attempts: {self.RECONNECT_ATTEMPTS}")
        
        # Display settings
        print(f"🖥️  Display: {self.DISPLAY_WIDTH}x{self.DISPLAY_HEIGHT}")
        print(f"   Max display: {self.MAX_DISPLAY_WIDTH}x{self.MAX_DISPLAY_HEIGHT}")
        
        # Detection settings
        print(f"🚗 Heavy traffic threshold: {self.HEAVY_TRAFFIC_THRESHOLD}")
        print(f"   Vehicle classes: {self.VEHICLE_CLASSES}")
        
        # Violence detection settings
        print(f"🛡️ Violence detection enabled: {self.VIOLENCE_DETECTION_ENABLED}")
        if self.VIOLENCE_DETECTION_ENABLED:
            print(f"   API User: {self.SIGHTENGINE_API_USER}")
            print(f"   API Secret: {'*' * len(self.SIGHTENGINE_API_SECRET)}")
            print(f"   Check interval: {self.VIOLENCE_CHECK_INTERVAL} frames")
            print(f"   Models: {self.VIOLENCE_MODELS}")
            print(f"   Threshold: {self.VIOLENCE_THRESHOLD}")
        
        print("=" * 50)

# Global configuration instance
config = Config()

# Preset configurations for different use cases
PRESETS = {
    'default': {
        'MODEL_CONFIDENCE': 0.4,
        'HEAVY_TRAFFIC_THRESHOLD': 8,
        'YOUTUBE_REFRESH_INTERVAL': 300
    },
    
    'high_accuracy': {
        'MODEL_CONFIDENCE': 0.6,
        'MODEL_IOU_THRESHOLD': 0.3,
        'HEAVY_TRAFFIC_THRESHOLD': 6
    },
    
    'performance': {
        'MODEL_CONFIDENCE': 0.3,
        'FRAME_SKIP': 1,
        'BUFFER_SIZE': 2,
        'MODEL_IMAGE_SIZE': 416
    },
    
    'server': {
        'MODEL_CONFIDENCE': 0.4,
        'FRAME_SKIP': 0,
        'STATISTICS_INTERVAL': 60,
        'MEMORY_CLEANUP_INTERVAL': 50
    },
    
    'demo': {
        'MODEL_CONFIDENCE': 0.3,
        'HEAVY_TRAFFIC_THRESHOLD': 5,
        'DISPLAY_WIDTH': 800,
        'DISPLAY_HEIGHT': 450
    }
}

def load_preset(preset_name: str):
    """Load a preset configuration"""
    if preset_name in PRESETS:
        preset = PRESETS[preset_name]
        for key, value in preset.items():
            if hasattr(config, key):
                setattr(config, key, value)
        print(f"✅ Loaded preset: {preset_name}")
    else:
        print(f"❌ Unknown preset: {preset_name}")
        print(f"Available presets: {list(PRESETS.keys())}")

def create_custom_config(**kwargs) -> Config:
    """Create custom configuration"""
    custom_config = Config()
    
    for key, value in kwargs.items():
        if hasattr(custom_config, key):
            setattr(custom_config, key, value)
        else:
            print(f"⚠️  Unknown config parameter: {key}")
    
    return custom_config

# Configuration validation functions
def validate_config() -> bool:
    """Validate current configuration"""
    print("🔍 Validating configuration...")
    
    valid = True
    
    # Check model file
    if not config.validate_model_path():
        valid = False
    
    # Check directories
    try:
        config.ensure_directories()
    except Exception as e:
        print(f"❌ Directory creation failed: {e}")
        valid = False
    
    # Validate thresholds
    if not 0.1 <= config.MODEL_CONFIDENCE <= 1.0:
        print(f"❌ Invalid confidence threshold: {config.MODEL_CONFIDENCE}")
        valid = False
    
    if config.HEAVY_TRAFFIC_THRESHOLD < 1:
        print(f"❌ Invalid traffic threshold: {config.HEAVY_TRAFFIC_THRESHOLD}")
        valid = False
    
    print(f"{'✅ Configuration valid' if valid else '❌ Configuration has issues'}")
    return valid

# System validation functions (merged from system_tools.py)
def check_system_requirements() -> bool:
    """Comprehensive system requirements check"""
    print("🔍 System Requirements Check")
    print("=" * 50)
    
    requirements_met = True
    
    # Python version
    python_version = sys.version_info
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    if python_version < (3, 8):
        print("   ❌ Python 3.8+ required")
        requirements_met = False
    else:
        print("   ✅ Python version OK")
    
    # OpenCV
    try:
        import cv2
        print(f"📹 OpenCV: {cv2.__version__}")
        
        # Check GUI support
        try:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("test")
            print("   ✅ GUI support available")
        except:
            print("   ⚠️  No GUI support (headless mode only)")
        
        # Check video codecs
        backends = []
        if hasattr(cv2, 'CAP_FFMPEG'):
            backends.append("FFMPEG")
        if hasattr(cv2, 'CAP_GSTREAMER'):
            backends.append("GStreamer")
        print(f"   📺 Video backends: {', '.join(backends) if backends else 'Default only'}")
        
    except ImportError:
        print("   ❌ OpenCV not installed")
        requirements_met = False
    
    # YOLO/Ultralytics
    try:
        from ultralytics import YOLO
        print("🎯 Ultralytics: Available")
        print("   ✅ YOLOv8 support OK")
    except ImportError:
        print("   ❌ Ultralytics not installed")
        requirements_met = False
    
    # yt-dlp for YouTube
    try:
        import yt_dlp
        print(f"📺 yt-dlp: Available")
        print("   ✅ YouTube support available")
    except ImportError:
        print("   ⚠️  yt-dlp not installed (YouTube streams disabled)")
    
    # NumPy
    try:
        import numpy as np
        print(f"🔢 NumPy: {np.__version__}")
        print("   ✅ NumPy OK")
    except ImportError:
        print("   ❌ NumPy not installed")
        requirements_met = False
    
    # Violence Detection API
    try:
        api_user = os.getenv('SIGHTENGINE_API_USER')
        api_secret = os.getenv('SIGHTENGINE_API_SECRET')
        
        if api_user and api_secret:
            print(f"🛡️ Sightengine API: Credentials available")
            print("   ✅ Violence detection ready")
        else:
            print("   ⚠️  No Sightengine credentials (violence detection disabled)")
    except Exception as e:
        print(f"   ❌ Violence detection error: {e}")
    
    print(f"{'✅ All requirements met' if requirements_met else '❌ Some requirements missing'}")
    return requirements_met

def test_violence_api():
    """Test violence detection API"""
    try:
        from violence_detector import create_violence_detector
        
        print("🧪 Testing violence detection API...")
        detector = create_violence_detector(config)
        
        if detector and detector.enabled:
            print("✅ Violence detection API test passed")
            return True
        else:
            print("❌ Violence detection not available")
            return False
    except Exception as e:
        print(f"❌ Violence detection test failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Management')
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    parser.add_argument('--check-system', action='store_true', help='Check system requirements')
    parser.add_argument('--test-api', action='store_true', help='Test violence detection API')
    parser.add_argument('--preset', type=str, help='Load preset configuration')
    parser.add_argument('--save', action='store_true', help='Save current configuration')
    parser.add_argument('--confidence', type=float, help='Set model confidence')
    parser.add_argument('--threshold', type=int, help='Set traffic threshold')
    
    args = parser.parse_args()
    
    if args.confidence:
        config.MODEL_CONFIDENCE = args.confidence
        print(f"✅ Confidence set to {args.confidence}")
    
    if args.threshold:
        config.HEAVY_TRAFFIC_THRESHOLD = args.threshold
        print(f"✅ Traffic threshold set to {args.threshold}")
    
    if args.preset:
        load_preset(args.preset)
    
    if args.check_system:
        check_system_requirements()
    
    if args.test_api:
        test_violence_api()
    
    if args.show or not any(vars(args).values()):
        config.print_config()
    
    if args.validate:
        validate_config()
    
    if args.save:
        config.save_user_config()
