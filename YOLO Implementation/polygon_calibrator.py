import cv2
import numpy as np

class PolygonCalibrator:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.points = []
        self.current_polygon = []
        self.polygons = []
        self.frame = None
        self.original_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append((x, y))
            print(f"Point added: ({x}, {y})")
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_polygon) >= 3:
                self.polygons.append(self.current_polygon.copy())
                print(f"Polygon {len(self.polygons)} completed with {len(self.current_polygon)} points")
                self.current_polygon = []
            else:
                print("Need at least 3 points for a polygon")
    
    def draw_polygons(self):
        self.frame = self.original_frame.copy()
        
        # Draw completed polygons
        for i, poly in enumerate(self.polygons):
            if len(poly) >= 3:
                pts = np.array(poly, dtype=np.int32)
                color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)  # Alternate green and blue
                cv2.polylines(self.frame, [pts], True, color, 3)
                
                # Label the polygon
                center = np.mean(pts, axis=0).astype(int)
                cv2.putText(self.frame, f"Lane {i+1}", tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw current polygon being created
        if len(self.current_polygon) > 0:
            for point in self.current_polygon:
                cv2.circle(self.frame, point, 5, (0, 255, 255), -1)
            
            if len(self.current_polygon) > 1:
                pts = np.array(self.current_polygon, dtype=np.int32)
                cv2.polylines(self.frame, [pts], False, (0, 255, 255), 2)
    
    def calibrate(self):
        if not self.cap.isOpened():
            print("Error: Could not open video")
            return None
        
        # Read first frame
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame")
            return None
        
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        
        print("Polygon Calibration Tool for Indian Traffic")
        print("=" * 50)
        print("Instructions:")
        print("1. Left-click to add points for a polygon")
        print("2. Right-click to complete current polygon")
        print("3. Create 2 polygons - one for each lane direction")
        print("4. Press 's' to save coordinates")
        print("5. Press 'r' to reset")
        print("6. Press 'q' to quit")
        print("=" * 50)
        
        cv2.namedWindow('Polygon Calibrator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Polygon Calibrator', self.mouse_callback)
        
        while True:
            self.draw_polygons()
            cv2.imshow('Polygon Calibrator', self.frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.polygons = []
                self.current_polygon = []
                print("Reset all polygons")
            elif key == ord('s'):
                self.save_coordinates()
        
        cv2.destroyAllWindows()
        self.cap.release()
        
        return self.polygons
    
    def save_coordinates(self):
        if len(self.polygons) < 2:
            print("Please create at least 2 polygons before saving")
            return
        
        print("\nGenerated coordinates for your video:")
        print("=" * 50)
        
        for i, poly in enumerate(self.polygons):
            print(f"vertices{i+1} = np.array({poly}, dtype=np.int32)")
        
        # Calculate lane threshold (approximately center between polygons)
        if len(self.polygons) >= 2:
            poly1_center_x = np.mean([p[0] for p in self.polygons[0]])
            poly2_center_x = np.mean([p[0] for p in self.polygons[1]])
            lane_threshold = int((poly1_center_x + poly2_center_x) / 2)
            print(f"lane_threshold = {lane_threshold}")
        
        # Suggest detection area based on polygons
        all_y_coords = []
        for poly in self.polygons:
            all_y_coords.extend([p[1] for p in poly])
        
        if all_y_coords:
            min_y = min(all_y_coords)
            max_y = max(all_y_coords)
            x1 = max(0, min_y - 50)  # Add some margin
            x2 = min(self.original_frame.shape[0], max_y + 50)
            print(f"x1, x2 = {x1}, {x2}  # Detection area")
        
        print("=" * 50)
        print("Copy these coordinates to your traffic analysis script!")
        
        # Save to file
        with open('polygon_coordinates.txt', 'w') as f:
            f.write("# Generated polygon coordinates for Indian traffic video\n")
            f.write("# Copy these to your traffic analysis script\n\n")
            
            for i, poly in enumerate(self.polygons):
                f.write(f"vertices{i+1} = np.array({poly}, dtype=np.int32)\n")
            
            if len(self.polygons) >= 2:
                f.write(f"lane_threshold = {lane_threshold}\n")
            
            if all_y_coords:
                f.write(f"x1, x2 = {x1}, {x2}  # Detection area\n")
        
        print("Coordinates saved to 'polygon_coordinates.txt'")

def main():
    video_path = input("Enter path to your Indian traffic video: ").strip().strip('"')
    
    if not video_path:
        video_path = "indian traffic.mp4"  # Default
    
    calibrator = PolygonCalibrator(video_path)
    polygons = calibrator.calibrate()
    
    if polygons:
        print(f"\nCalibration completed! Generated {len(polygons)} polygons.")
    else:
        print("Calibration cancelled.")

if __name__ == "__main__":
    main()
