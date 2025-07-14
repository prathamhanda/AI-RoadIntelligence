#!/usr/bin/env python3
"""
🔧 System Tools and Utilities for Traffic Analysis System
System checking, debugging, and testing utilities

Author: Pratham Handa
GitHub: https://github.com/prathamhanda/IoT-Based_Traffic_Regulation
"""

import cv2
import os
import sys
import time
import json
import subprocess
import platform
from datetime import datetime
import numpy as np

def check_system_requirements():
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
        if cv2.CAP_FFMPEG in dir(cv2):
            backends.append("FFMPEG")
        if cv2.CAP_GSTREAMER in dir(cv2):
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
        print(f"📺 yt-dlp: {yt_dlp.version.__version__}")
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
    
    # System info
    print(f"\n💻 System: {platform.system()} {platform.release()}")
    print(f"🏗️  Architecture: {platform.machine()}")
    
    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"🚀 GPU: {gpu_name} ({gpu_count} devices)")
            print("   ✅ CUDA acceleration available")
        else:
            print("   ⚠️  No CUDA GPU detected (CPU mode)")
    except ImportError:
        print("   ⚠️  PyTorch not installed (CPU mode)")
    
    print("\n" + "=" * 50)
    if requirements_met:
        print("✅ All critical requirements met!")
    else:
        print("❌ Some requirements missing - check installation")
    
    return requirements_met

def test_video_sources():
    """Test different video input sources"""
    print("\n🎥 Video Source Compatibility Test")
    print("=" * 50)
    
    test_results = {}
    
    # Test webcam
    print("📷 Testing webcam (camera 0)...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"   ✅ Webcam: {width}x{height}")
                test_results['webcam'] = True
            else:
                print("   ❌ Webcam: Cannot read frames")
                test_results['webcam'] = False
        else:
            print("   ❌ Webcam: Cannot open")
            test_results['webcam'] = False
        cap.release()
    except Exception as e:
        print(f"   ❌ Webcam error: {e}")
        test_results['webcam'] = False
    
    # Test video file
    print("\n📁 Testing video file support...")
    test_video_path = "test_video.mp4"
    if os.path.exists(test_video_path):
        try:
            cap = cv2.VideoCapture(test_video_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    height, width = frame.shape[:2]
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print(f"   ✅ Video file: {width}x{height}, {total_frames} frames, {fps:.1f} FPS")
                    test_results['video_file'] = True
                else:
                    print("   ❌ Video file: Cannot read frames")
                    test_results['video_file'] = False
            else:
                print("   ❌ Video file: Cannot open")
                test_results['video_file'] = False
            cap.release()
        except Exception as e:
            print(f"   ❌ Video file error: {e}")
            test_results['video_file'] = False
    else:
        print(f"   ⚠️  No test video file found ({test_video_path})")
        test_results['video_file'] = None
    
    # Test HTTP stream (if URL provided)
    test_stream_url = "http://sample-videos.com/zip/10/mp4/SampleVideo_360x240_1mb.mp4"
    print(f"\n🌐 Testing HTTP stream...")
    try:
        cap = cv2.VideoCapture(test_stream_url)
        if cap.isOpened():
            # Set timeout
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 2000)
            
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"   ✅ HTTP stream: {width}x{height}")
                test_results['http_stream'] = True
            else:
                print("   ❌ HTTP stream: Cannot read frames")
                test_results['http_stream'] = False
        else:
            print("   ❌ HTTP stream: Cannot connect")
            test_results['http_stream'] = False
        cap.release()
    except Exception as e:
        print(f"   ⚠️  HTTP stream test skipped: {e}")
        test_results['http_stream'] = None
    
    # Test YouTube (if yt-dlp available)
    try:
        import yt_dlp
        print(f"\n📺 Testing YouTube support...")
        
        test_youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Short test video
        
        ydl_opts = {
            'format': 'best[height<=360]',
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(test_youtube_url, download=False)
                if info and info.get('url'):
                    print("   ✅ YouTube URL extraction working")
                    test_results['youtube'] = True
                else:
                    print("   ❌ YouTube URL extraction failed")
                    test_results['youtube'] = False
        except Exception as e:
            print(f"   ❌ YouTube test failed: {e}")
            test_results['youtube'] = False
            
    except ImportError:
        print("   ⚠️  yt-dlp not available")
        test_results['youtube'] = None
    
    print(f"\n📊 Test Summary:")
    for source, result in test_results.items():
        status = "✅ PASS" if result is True else "❌ FAIL" if result is False else "⚠️  SKIP"
        print(f"   {source}: {status}")
    
    return test_results

def test_yolo_model(model_path="models/best.pt"):
    """Test YOLO model loading and inference"""
    print(f"\n🎯 YOLO Model Test: {model_path}")
    print("=" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        from ultralytics import YOLO
        
        print("📦 Loading model...")
        model = YOLO(model_path)
        print(f"   ✅ Model loaded successfully")
        print(f"   📁 Model size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        
        # Create test image
        print("\n🖼️  Creating test image...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test inference
        print("🔍 Running test inference...")
        start_time = time.time()
        results = model.predict(test_image, verbose=False)
        inference_time = time.time() - start_time
        
        print(f"   ✅ Inference successful")
        print(f"   ⏱️  Inference time: {inference_time:.3f} seconds")
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                num_detections = len(result.boxes)
                print(f"   📊 Detections: {num_detections}")
            else:
                print(f"   📊 No detections (expected for random image)")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_stream_connection(url, timeout=10):
    """Test connection to a specific stream URL"""
    print(f"\n🔗 Testing stream connection: {url}")
    print("=" * 50)
    
    try:
        # Handle YouTube URLs
        if 'youtube.com' in url or 'youtu.be' in url:
            try:
                import yt_dlp
                print("📺 YouTube URL detected, extracting stream...")
                
                ydl_opts = {
                    'format': 'best[height<=720]',
                    'quiet': True,
                    'no_warnings': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    if info and info.get('url'):
                        stream_url = info['url']
                        print(f"   ✅ Stream URL extracted")
                        print(f"   📺 Title: {info.get('title', 'Unknown')}")
                        if info.get('is_live'):
                            print(f"   🔴 Live stream detected")
                        url = stream_url
                    else:
                        print("   ❌ Could not extract stream URL")
                        return False
            except ImportError:
                print("   ❌ yt-dlp required for YouTube streams")
                return False
            except Exception as e:
                print(f"   ❌ YouTube extraction failed: {e}")
                return False
        
        # Test OpenCV connection
        print("🔌 Testing OpenCV connection...")
        backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
        
        for i, backend in enumerate(backends):
            print(f"   🔄 Trying backend {i+1}/{len(backends)}...")
            try:
                cap = cv2.VideoCapture(url, backend)
                
                if cap.isOpened():
                    # Set timeouts
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Test frame reading
                    start_time = time.time()
                    ret, frame = cap.read()
                    read_time = time.time() - start_time
                    
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"   ✅ Connection successful!")
                        print(f"   📐 Resolution: {width}x{height}")
                        print(f"   ⏱️  Frame read time: {read_time:.3f}s")
                        
                        # Test multiple frames
                        successful_reads = 0
                        for _ in range(5):
                            ret, _ = cap.read()
                            if ret:
                                successful_reads += 1
                        
                        print(f"   📊 Frame stability: {successful_reads}/5 frames")
                        cap.release()
                        return True
                    else:
                        print(f"   ❌ Cannot read frames (backend {i+1})")
                
                cap.release()
                
            except Exception as e:
                print(f"   ❌ Backend {i+1} error: {e}")
        
        print("   ❌ All backends failed")
        return False
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def benchmark_system():
    """Run performance benchmarks"""
    print("\n🚀 Performance Benchmark")
    print("=" * 50)
    
    # OpenCV performance test
    print("📹 OpenCV frame processing benchmark...")
    test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    # Resize benchmark
    start_time = time.time()
    for _ in range(100):
        resized = cv2.resize(test_image, (640, 360))
    resize_time = time.time() - start_time
    print(f"   Resize (100x): {resize_time:.3f}s ({100/resize_time:.1f} FPS)")
    
    # Color conversion benchmark
    start_time = time.time()
    for _ in range(100):
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    color_time = time.time() - start_time
    print(f"   Color convert (100x): {color_time:.3f}s ({100/color_time:.1f} FPS)")
    
    # YOLO inference benchmark (if available)
    try:
        from ultralytics import YOLO
        if os.path.exists("models/best.pt"):
            print("\n🎯 YOLO inference benchmark...")
            model = YOLO("models/best.pt")
            test_img = cv2.resize(test_image, (640, 640))
            
            # Warmup
            for _ in range(3):
                _ = model.predict(test_img, verbose=False)
            
            # Benchmark
            start_time = time.time()
            for _ in range(10):
                results = model.predict(test_img, verbose=False)
            inference_time = time.time() - start_time
            print(f"   YOLO inference (10x): {inference_time:.3f}s ({10/inference_time:.1f} FPS)")
    except:
        print("   ⚠️  YOLO benchmark skipped")
    
    # Memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\n💾 Memory: {memory.percent:.1f}% used ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        
        cpu_percent = psutil.cpu_percent(interval=1)
        print(f"🖥️  CPU: {cpu_percent:.1f}% usage")
    except ImportError:
        print("   ⚠️  psutil not available for system stats")

def save_diagnostic_report():
    """Save comprehensive diagnostic report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'system': {
            'platform': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        },
        'requirements_check': None,
        'video_sources': None,
        'model_test': None
    }
    
    print("\n📋 Generating diagnostic report...")
    
    # Run all tests
    report['requirements_check'] = check_system_requirements()
    report['video_sources'] = test_video_sources()
    
    if os.path.exists("models/best.pt"):
        report['model_test'] = test_yolo_model()
    
    # Save report
    report_file = f"diagnostic_report_{int(time.time())}.json"
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Diagnostic report saved: {report_file}")
    except Exception as e:
        print(f"❌ Could not save report: {e}")
    
    return report

def main():
    """Main system tools interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Traffic Analysis System Tools')
    parser.add_argument('--check', action='store_true', 
                       help='Check system requirements')
    parser.add_argument('--test-sources', action='store_true', 
                       help='Test video input sources')
    parser.add_argument('--test-model', type=str, default='models/best.pt',
                       help='Test YOLO model')
    parser.add_argument('--test-stream', type=str, 
                       help='Test specific stream URL')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmarks')
    parser.add_argument('--full-report', action='store_true', 
                       help='Generate full diagnostic report')
    
    args = parser.parse_args()
    
    print("🔧 Traffic Analysis System Tools")
    print("=" * 60)
    
    if args.check or not any(vars(args).values()):
        check_system_requirements()
    
    if args.test_sources:
        test_video_sources()
    
    if args.test_model:
        test_yolo_model(args.test_model)
    
    if args.test_stream:
        test_stream_connection(args.test_stream)
    
    if args.benchmark:
        benchmark_system()
    
    if args.full_report:
        save_diagnostic_report()

if __name__ == "__main__":
    main()
