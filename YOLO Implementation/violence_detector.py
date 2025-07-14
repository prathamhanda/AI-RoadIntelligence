#!/usr/bin/env python3
"""
🛡️ Violence Detection Module for Traffic Analysis System
Advanced content moderation using Sightengine API to detect violence, gore, and weapons in traffic footage

Features:
- Real-time violence detection using Sightengine API
- Configurable detection models (violence, gore, weapons)
- Evidence saving and logging
- Alert system integration
- Asynchronous processing for performance

Author: Pratham Handa
GitHub: https://github.com/prathamhanda/IoT-Based_Traffic_Regulation
"""

import cv2
import numpy as np
import requests
import json
import time
import threading
from queue import Queue, Empty
from datetime import datetime
import os
import logging
from typing import Dict, List, Tuple, Optional, Any

class ViolenceDetector:
    """Violence detection using Sightengine API with async processing"""
    
    def __init__(self, config):
        self.config = config
        self.api_user = config.SIGHTENGINE_API_USER
        self.api_secret = config.SIGHTENGINE_API_SECRET
        self.enabled = config.VIOLENCE_DETECTION_ENABLED
        self.check_interval = config.VIOLENCE_CHECK_INTERVAL
        self.models = config.VIOLENCE_MODELS
        self.threshold = config.VIOLENCE_THRESHOLD
        
        # Processing queues
        self.frame_queue = Queue(maxsize=5)
        self.result_queue = Queue(maxsize=10)
        
        # Detection state
        self.frame_count = 0
        self.last_check = 0
        self.processing = False
        self.worker_thread = None
        
        # Statistics
        self.stats = {
            'total_checks': 0,
            'violence_detected': 0,
            'api_errors': 0,
            'last_detection': None,
            'processing_time': 0
        }
        
        # Evidence storage
        self.evidence_dir = "evidence"
        if config.VIOLENCE_SAVE_EVIDENCE:
            self.ensure_evidence_dir()
        
        # Setup logging
        self.setup_logging()
        
        if self.enabled:
            self.logger.info("[SHIELD] Violence detector initialized")
            self.logger.info(f"Models: {', '.join(self.models)}")
            self.logger.info(f"Threshold: {self.threshold}")
        else:
            self.logger.warning("[WARNING] Violence detection disabled - no API credentials")
    
    def setup_logging(self):
        """Setup logging for violence detection"""
        self.logger = logging.getLogger('ViolenceDetector')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # Create logs directory if it doesn't exist
            os.makedirs('logs', exist_ok=True)
            
            # File handler
            fh = logging.FileHandler('logs/violence_detection.log')
            fh.setLevel(logging.INFO)
            
            # Console handler
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
    
    def ensure_evidence_dir(self):
        """Create evidence directory structure"""
        try:
            os.makedirs(self.evidence_dir, exist_ok=True)
            os.makedirs(os.path.join(self.evidence_dir, 'violence'), exist_ok=True)
            os.makedirs(os.path.join(self.evidence_dir, 'gore'), exist_ok=True)
            os.makedirs(os.path.join(self.evidence_dir, 'weapons'), exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create evidence directories: {e}")
    
    def start_processing(self):
        """Start background processing thread"""
        if not self.enabled or self.processing:
            return
        
        self.processing = True
        self.worker_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.worker_thread.start()
        self.logger.info("[ROCKET] Violence detection processing started")
    
    def stop_processing(self):
        """Stop background processing"""
        self.processing = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.logger.info("[STOP] Violence detection processing stopped")
    
    def should_check_frame(self) -> bool:
        """Determine if current frame should be checked"""
        self.frame_count += 1
        
        # Check based on interval
        if self.frame_count - self.last_check >= self.check_interval:
            return True
        
        return False
    
    def queue_frame(self, frame: np.ndarray) -> bool:
        """Queue frame for violence detection"""
        if not self.enabled or not self.should_check_frame():
            return False
        
        try:
            # Convert frame to required format
            processed_frame = self._preprocess_frame(frame)
            self.frame_queue.put_nowait((self.frame_count, processed_frame))
            self.last_check = self.frame_count
            return True
        except:
            return False
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for API submission"""
        # Resize if too large (API limits)
        height, width = frame.shape[:2]
        max_dimension = 1280
        
        if max(height, width) > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * max_dimension / width)
            else:
                new_height = max_dimension
                new_width = int(width * max_dimension / height)
            
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def _process_frames(self):
        """Background frame processing worker"""
        while self.processing:
            try:
                # Get frame from queue
                frame_id, frame = self.frame_queue.get(timeout=1)
                
                # Process frame
                result = self._analyze_frame(frame_id, frame)
                
                if result:
                    self.result_queue.put(result)
                
            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"Frame processing error: {e}")
                self.stats['api_errors'] += 1
    
    def _analyze_frame(self, frame_id: int, frame: np.ndarray) -> Optional[Dict]:
        """Analyze frame using Sightengine API"""
        start_time = time.time()
        
        try:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # Prepare API request
            files = {'media': buffer.tobytes()}
            data = {
                'models': ','.join(self.models),
                'api_user': self.api_user,
                'api_secret': self.api_secret
            }
            
            # Make API request
            response = requests.post(
                'https://api.sightengine.com/1.0/check.json',
                files=files,
                data=data,
                timeout=10
            )
            
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            self.stats['total_checks'] += 1
            
            if response.status_code == 200:
                result = response.json()
                return self._process_api_response(frame_id, frame, result, processing_time)
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                self.stats['api_errors'] += 1
                return None
                
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            self.stats['api_errors'] += 1
            return None
    
    def _process_api_response(self, frame_id: int, frame: np.ndarray, 
                            api_result: Dict, processing_time: float) -> Optional[Dict]:
        """Process API response and determine if violence detected"""
        violence_detected = False
        detections = {}
        max_confidence = 0
        
        # Check each model result
        for model in self.models:
            if model in api_result:
                confidence = api_result[model].get('prob', 0)
                detections[model] = confidence
                
                if confidence > self.threshold:
                    violence_detected = True
                    max_confidence = max(max_confidence, confidence)
        
        if violence_detected:
            self.stats['violence_detected'] += 1
            self.stats['last_detection'] = datetime.now()
            
            # Create detection result
            detection_result = {
                'frame_id': frame_id,
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'max_confidence': max_confidence,
                'processing_time': processing_time,
                'frame_shape': frame.shape
            }
            
            # Log detection
            self.logger.warning(
                f"[ALERT] VIOLENCE DETECTED - Frame {frame_id}: "
                f"{max_confidence:.2f} confidence"
            )
            
            # Save evidence if enabled
            if self.config.VIOLENCE_SAVE_EVIDENCE:
                self._save_evidence(frame_id, frame, detection_result)
            
            return detection_result
        
        return None
    
    def _save_evidence(self, frame_id: int, frame: np.ndarray, detection: Dict):
        """Save evidence frame with detection overlay"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Determine primary detection type
            detections = detection['detections']
            primary_type = max(detections.items(), key=lambda x: x[1])[0]
            
            # Create filename
            filename = f"evidence_{timestamp}_frame_{frame_id}_{primary_type}.jpg"
            filepath = os.path.join(self.evidence_dir, primary_type, filename)
            
            # Add detection overlay
            evidence_frame = self._add_detection_overlay(frame.copy(), detection)
            
            # Save frame
            cv2.imwrite(filepath, evidence_frame)
            
            # Save detection metadata
            metadata_file = filepath.replace('.jpg', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(detection, f, indent=2)
            
            self.logger.info(f"💾 Evidence saved: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save evidence: {e}")
    
    def _add_detection_overlay(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """Add detection information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Add semi-transparent red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 80), (0, 0, 255), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Title
        cv2.putText(frame, "[!] VIOLENCE DETECTED", (10, 25), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Detection details
        y_offset = 50
        for model, confidence in detection['detections'].items():
            if confidence > self.threshold:
                text = f"{model.upper()}: {confidence:.2f}"
                cv2.putText(frame, text, (10, y_offset), 
                           font, font_scale * 0.8, (255, 255, 255), thickness)
                y_offset += 25
        
        # Timestamp
        timestamp = detection['timestamp'][:19]  # Remove microseconds
        cv2.putText(frame, timestamp, (width - 200, height - 10), 
                   font, font_scale * 0.6, (255, 255, 255), thickness)
        
        return frame
    
    def get_latest_detection(self) -> Optional[Dict]:
        """Get latest violence detection result"""
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def get_statistics(self) -> Dict:
        """Get violence detection statistics"""
        avg_processing_time = (
            self.stats['processing_time'] / max(1, self.stats['total_checks'])
        )
        
        return {
            'enabled': self.enabled,
            'total_checks': self.stats['total_checks'],
            'violence_detected': self.stats['violence_detected'],
            'api_errors': self.stats['api_errors'],
            'detection_rate': self.stats['violence_detected'] / max(1, self.stats['total_checks']),
            'avg_processing_time': avg_processing_time,
            'last_detection': self.stats['last_detection'].isoformat() if self.stats['last_detection'] else None,
            'queue_size': self.frame_queue.qsize(),
            'pending_results': self.result_queue.qsize()
        }
    
    def create_alert_overlay(self, frame: np.ndarray, detection: Dict) -> np.ndarray:
        """Create visual alert overlay for live display"""
        height, width = frame.shape[:2]
        
        # Create flashing red border
        border_thickness = 10
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), border_thickness)
        
        # Add alert text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(width / 800, height / 600) * 1.5
        thickness = 3
        
        # Alert message
        alert_text = "[!] VIOLENCE DETECTED [!]"
        text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = 60
        
        # Background for text
        cv2.rectangle(frame, (text_x - 10, text_y - 40), 
                     (text_x + text_size[0] + 10, text_y + 10), 
                     (0, 0, 0), -1)
        
        # Alert text
        cv2.putText(frame, alert_text, (text_x, text_y), 
                   font, font_scale, (0, 0, 255), thickness)
        
        # Detection confidence
        max_conf = detection['max_confidence']
        conf_text = f"Confidence: {max_conf:.1%}"
        cv2.putText(frame, conf_text, (text_x, text_y + 50), 
                   font, font_scale * 0.7, (255, 255, 255), thickness - 1)
        
        return frame
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_processing()


class ViolenceAlertManager:
    """Manages violence detection alerts and notifications"""
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.VIOLENCE_ALERT_ENABLED
        self.alerts = []
        self.max_alerts = 100
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup alert logging"""
        self.logger = logging.getLogger('ViolenceAlerts')
        self.logger.setLevel(logging.WARNING)
        
        if not self.logger.handlers:
            os.makedirs('logs', exist_ok=True)
            
            fh = logging.FileHandler('logs/violence_alerts.log')
            fh.setLevel(logging.WARNING)
            
            formatter = logging.Formatter(
                '%(asctime)s - ALERT - %(message)s'
            )
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
    
    def process_detection(self, detection: Dict):
        """Process violence detection and create alerts"""
        if not self.enabled:
            return
        
        alert = {
            'id': len(self.alerts) + 1,
            'timestamp': datetime.now(),
            'detection': detection,
            'severity': self._calculate_severity(detection),
            'acknowledged': False
        }
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
        
        # Log alert
        self.logger.warning(
            f"Violence Alert #{alert['id']}: "
            f"Severity {alert['severity']}, "
            f"Max confidence {detection['max_confidence']:.2f}"
        )
        
        # Additional alert actions can be added here
        # (email, SMS, webhook, etc.)
    
    def _calculate_severity(self, detection: Dict) -> str:
        """Calculate alert severity based on detection confidence"""
        max_conf = detection['max_confidence']
        
        if max_conf >= 0.9:
            return "CRITICAL"
        elif max_conf >= 0.8:
            return "HIGH"
        elif max_conf >= 0.7:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return self.alerts[-count:] if self.alerts else []
    
    def acknowledge_alert(self, alert_id: int):
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                break
    
    def get_unacknowledged_count(self) -> int:
        """Get count of unacknowledged alerts"""
        return sum(1 for alert in self.alerts if not alert['acknowledged'])


# Factory function for easy integration
def create_violence_detector(config) -> Optional[ViolenceDetector]:
    """Create violence detector instance if enabled"""
    if config.VIOLENCE_DETECTION_ENABLED:
        return ViolenceDetector(config)
    return None

def create_alert_manager(config) -> ViolenceAlertManager:
    """Create alert manager instance"""
    return ViolenceAlertManager(config)
