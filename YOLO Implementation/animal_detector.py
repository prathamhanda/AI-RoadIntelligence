"""
Animal Detection Module for Traffic Analysis System
Detects animals in traffic streams and saves evidence of incidents
"""

import cv2
import numpy as np
import os
import json
import csv
from datetime import datetime, timedelta
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional
import asyncio

class AnimalDetector:
    """
    Handles animal detection using YOLOv8 model
    """
    
    def __init__(self, model_path: str = "Animal_detection_grouped/yolov8n.pt", 
                 confidence_threshold: float = 0.25):
        """
        Initialize animal detector
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Minimum confidence for detection (lowered to match standalone detection)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        # Target animals for traffic detection
        self.target_animals = {
            'cow': 'cattle',
            'horse': 'horse',
            'dog': 'dog',
            'cat': 'cat',
            'sheep': 'sheep',
            'bird': 'bird',
            'elephant': 'elephant'
        }
        
        # COCO class mapping for animals (corrected based on actual YOLOv8 COCO classes)
        self.coco_animal_classes = {
            14: 'bird',    # bird  
            15: 'cat',     # cat
            16: 'dog',     # dog
            17: 'horse',   # horse
            18: 'sheep',   # sheep
            19: 'cow',     # cow
            20: 'elephant' # elephant
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model"""
        try:
            # Try the specified path first
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"✅ Animal detection model loaded: {self.model_path}")
            # Try relative to current working directory
            elif os.path.exists(os.path.join("..", self.model_path)):
                model_path = os.path.join("..", self.model_path)
                self.model = YOLO(model_path)
                print(f"✅ Animal detection model loaded: {model_path}")
            # Try in the Animal_detection_grouped folder
            elif os.path.exists("../Animal_detection_grouped/yolov8n.pt"):
                self.model = YOLO("../Animal_detection_grouped/yolov8n.pt")
                print(f"✅ Animal detection model loaded: ../Animal_detection_grouped/yolov8n.pt")
            else:
                print(f"⚠️ Model not found at {self.model_path}, downloading YOLOv8n")
                self.model = YOLO('yolov8n.pt')
                print("✅ Downloaded and loaded YOLOv8n model")
            
            # Set model to CPU (avoid CUDA issues)
            try:
                self.model.to('cpu')
                print("🎯 Model running on: CPU")
            except Exception as device_error:
                print(f"⚠️ Device setting warning: {device_error}")
                print("🎯 Model running on: default device")
            
        except Exception as e:
            print(f"❌ Error loading animal detection model: {e}")
            # Try fallback to basic YOLOv8n
            try:
                print("🔄 Trying fallback to YOLOv8n...")
                self.model = YOLO('yolov8n.pt')
                self.model.to('cpu')
                print("✅ Fallback model loaded successfully")
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")
                self.model = None
    
    def detect_animals(self, frame: np.ndarray) -> Tuple[Dict[str, int], List[Dict]]:
        """
        Detect animals in frame
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (animal_counts, detections)
        """
        if self.model is None:
            return {}, []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            animal_counts = {animal: 0 for animal in self.target_animals.keys()}
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID and confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Check if it's an animal we're interested in
                        if class_id in self.coco_animal_classes:
                            animal_type = self.coco_animal_classes[class_id]
                            
                            if animal_type in self.target_animals:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
                                animal_counts[animal_type] += 1
                                
                                detections.append({
                                    'type': animal_type,
                                    'confidence': confidence,
                                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                    'class_id': class_id
                                })
            
            return animal_counts, detections
            
        except Exception as e:
            print(f"❌ Error in animal detection: {e}")
            return {}, []
    
    def annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Annotate frame with animal detections
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            animal_type = detection['type']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            
            # Draw bounding box
            color = (0, 255, 255)  # Yellow for animals
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{animal_type.upper()}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for label
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            
            # Label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return annotated_frame


class AnimalAlertManager:
    """
    Manages animal detection alerts and evidence saving
    """
    
    def __init__(self, evidence_dir: str = "evidence/animals"):
        """
        Initialize alert manager
        
        Args:
            evidence_dir: Directory to save evidence
        """
        self.evidence_dir = evidence_dir
        self.csv_file_path = os.path.join(evidence_dir, "animal_detections.csv")
        self.alerts_file_path = os.path.join(evidence_dir, "animal_alerts.json")
        
        # Create evidence directory
        os.makedirs(evidence_dir, exist_ok=True)
        
        # Initialize CSV file
        self._init_csv_file()
        
        # Load existing alerts
        self.alerts = self._load_alerts()
        
        # Alert cooldown (seconds)
        self.alert_cooldown = 30
        self.last_alert_time = {}
    
    def _init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist"""
        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'timestamp', 'frame_number', 'animal_type', 'count', 
                    'confidence_avg', 'evidence_image', 'coordinates'
                ])
    
    def _load_alerts(self) -> List[Dict]:
        """Load existing alerts from JSON file"""
        if os.path.exists(self.alerts_file_path):
            try:
                with open(self.alerts_file_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except Exception as e:
                print(f"⚠️ Error loading animal alerts: {e}")
        return []
    
    def _save_alerts(self):
        """Save alerts to JSON file"""
        try:
            with open(self.alerts_file_path, 'w', encoding='utf-8') as file:
                json.dump(self.alerts, file, indent=2, default=str)
        except Exception as e:
            print(f"❌ Error saving animal alerts: {e}")
    
    async def save_evidence(self, frame: np.ndarray, animal_counts: Dict[str, int], 
                          detections: List[Dict], frame_number: int) -> Optional[str]:
        """
        Save evidence of animal detection
        
        Args:
            frame: Current frame
            animal_counts: Dictionary of animal counts
            detections: List of detection data
            frame_number: Current frame number
            
        Returns:
            Path to saved evidence image
        """
        try:
            timestamp = datetime.now()
            
            # Create filename
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            evidence_filename = f"animal_detection_{timestamp_str}_frame_{frame_number}.jpg"
            evidence_path = os.path.join(self.evidence_dir, evidence_filename)
            
            # Annotate frame with overlay
            evidence_frame = self._create_evidence_overlay(frame, animal_counts, detections, timestamp)
            
            # Save evidence image
            cv2.imwrite(evidence_path, evidence_frame)
            
            # Log to CSV
            await self._log_to_csv(timestamp, frame_number, animal_counts, detections, evidence_filename)
            
            print(f"📁 Animal evidence saved: {evidence_filename}")
            return evidence_path
            
        except Exception as e:
            print(f"❌ Error saving animal evidence: {e}")
            return None
    
    def _create_evidence_overlay(self, frame: np.ndarray, animal_counts: Dict[str, int], 
                               detections: List[Dict], timestamp: datetime) -> np.ndarray:
        """
        Create evidence frame with overlay information
        
        Args:
            frame: Original frame
            animal_counts: Animal counts
            detections: Detection data
            timestamp: Detection timestamp
            
        Returns:
            Frame with evidence overlay
        """
        evidence_frame = frame.copy()
        
        # Add timestamp
        timestamp_text = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(evidence_frame, f"TIMESTAMP: {timestamp_text}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add alert header
        cv2.putText(evidence_frame, "ANIMAL DETECTION ALERT", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add animal counts
        y_offset = 90
        for animal, count in animal_counts.items():
            if count > 0:
                text = f"{animal.upper()}: {count} detected"
                cv2.putText(evidence_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_offset += 25
        
        return evidence_frame
    
    async def _log_to_csv(self, timestamp: datetime, frame_number: int, 
                         animal_counts: Dict[str, int], detections: List[Dict], 
                         evidence_filename: str):
        """
        Log detection to CSV file
        
        Args:
            timestamp: Detection timestamp
            frame_number: Frame number
            animal_counts: Animal counts
            detections: Detection data
            evidence_filename: Evidence file name
        """
        try:
            with open(self.csv_file_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for animal, count in animal_counts.items():
                    if count > 0:
                        # Calculate average confidence for this animal type
                        animal_detections = [d for d in detections if d['type'] == animal]
                        avg_confidence = sum(d['confidence'] for d in animal_detections) / len(animal_detections)
                        
                        # Get coordinates
                        coordinates = [d['bbox'] for d in animal_detections]
                        
                        # Write row
                        row = [
                            timestamp.isoformat(),
                            frame_number,
                            animal,
                            count,
                            f"{avg_confidence:.2f}",
                            evidence_filename,
                            str(coordinates)
                        ]
                        
                        writer.writerow(row)
                        
        except Exception as e:
            print(f"❌ Error logging to CSV: {e}")
    
    def should_alert(self, animal_type: str) -> bool:
        """
        Check if we should send an alert for this animal type
        
        Args:
            animal_type: Type of animal detected
            
        Returns:
            True if alert should be sent
        """
        current_time = datetime.now()
        
        if animal_type not in self.last_alert_time:
            self.last_alert_time[animal_type] = current_time
            return True
        
        time_since_last = (current_time - self.last_alert_time[animal_type]).total_seconds()
        
        if time_since_last >= self.alert_cooldown:
            self.last_alert_time[animal_type] = current_time
            return True
        
        return False
    
    async def create_alert(self, animal_counts: Dict[str, int], detections: List[Dict], 
                          frame_number: int, evidence_path: str) -> Dict:
        """
        Create an alert for animal detection
        
        Args:
            animal_counts: Animal counts
            detections: Detection data
            frame_number: Frame number
            evidence_path: Path to evidence image
            
        Returns:
            Alert dictionary
        """
        alert = {
            'id': f"animal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{frame_number}",
            'timestamp': datetime.now().isoformat(),
            'type': 'animal_detection',
            'frame_number': frame_number,
            'animal_counts': animal_counts,
            'total_animals': sum(animal_counts.values()),
            'evidence_path': evidence_path,
            'detections': detections,
            'severity': self._calculate_severity(animal_counts),
            'location': 'traffic_stream'
        }
        
        self.alerts.append(alert)
        self._save_alerts()
        
        return alert
    
    def _calculate_severity(self, animal_counts: Dict[str, int]) -> str:
        """
        Calculate severity level based on animal counts
        
        Args:
            animal_counts: Dictionary of animal counts
            
        Returns:
            Severity level string
        """
        total_animals = sum(animal_counts.values())
        
        # Check for dangerous animals
        dangerous_animals = ['elephant', 'cow']  # Large animals that pose traffic risk
        has_dangerous = any(animal_counts.get(animal, 0) > 0 for animal in dangerous_animals)
        
        if total_animals >= 5 or has_dangerous:
            return 'HIGH'
        elif total_animals >= 3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """
        Get recent alerts within specified hours
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = []
        for alert in self.alerts:
            try:
                alert_time = datetime.fromisoformat(alert['timestamp'])
                if alert_time >= cutoff_time:
                    recent_alerts.append(alert)
            except:
                continue
        
        return sorted(recent_alerts, key=lambda x: x['timestamp'], reverse=True)
