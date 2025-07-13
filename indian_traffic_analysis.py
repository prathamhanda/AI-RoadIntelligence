import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Vehicle Detection and Traffic Analysis - Indian Traffic Optimized')
    parser.add_argument('--source', type=str, default='sample_video.mp4', 
                       help='Path to input video file')
    parser.add_argument('--weights', type=str, default='models/best.pt', 
                       help='Path to model weights')
    parser.add_argument('--output', type=str, default='processed_sample_video.avi', 
                       help='Path to output video file')
    parser.add_argument('--conf', type=float, default=0.4, 
                       help='Confidence threshold for detection')
    parser.add_argument('--webcam', action='store_true', 
                       help='Use webcam instead of video file')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.weights):
        print(f"Error: Model weights file '{args.weights}' not found!")
        print("Please ensure the model file exists in the models/ directory.")
        return
    
    # Load the best fine-tuned YOLOv8 model
    try:
        best_model = YOLO(args.weights)
        print(f"Successfully loaded model: {args.weights}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Define the threshold for considering traffic as heavy
    heavy_traffic_threshold = 8  # Reduced for Indian traffic conditions

    # Initialize video capture first
    if args.webcam:
        cap = cv2.VideoCapture(0)  # Use default webcam
        print("Using webcam...")
    else:
        if not os.path.exists(args.source):
            print(f"Error: Video file '{args.source}' not found!")
            print("Please provide a valid video file path or use --webcam for live feed.")
            return
        cap = cv2.VideoCapture(args.source)
        print(f"Processing video: {args.source}")
    
    if not cap.isOpened():
        print("Error: Could not open video source!")
        return

    # Get video dimensions
    ret, sample_frame = cap.read()
    if not ret:
        print("Error: Could not read sample frame for calibration")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
    
    frame_height, frame_width = sample_frame.shape[:2]
    print(f"Video dimensions: {frame_width}x{frame_height}")
    
    # Calculate display scale to fit screen (assuming max screen width of 1920)
    max_display_width = 1280  # Reasonable size for most screens
    max_display_height = 720
    
    if frame_width > max_display_width or frame_height > max_display_height:
        width_scale = max_display_width / frame_width
        height_scale = max_display_height / frame_height
        display_scale = min(width_scale, height_scale)
        display_width = int(frame_width * display_scale)
        display_height = int(frame_height * display_scale)
        print(f"Display scaled to: {display_width}x{display_height} (scale: {display_scale:.2f})")
    else:
        display_scale = 1.0
        display_width = frame_width
        display_height = frame_height
        print("Using original video dimensions for display")

    # CUSTOM COORDINATES: Generated from your Indian traffic video calibration
    # These are the exact coordinates you marked for your specific video
    vertices1 = np.array([(192, 2152), (2048, 926), (2835, 926), (3125, 2155), (190, 2152)], dtype=np.int32)
    vertices2 = np.array([(2, 1711), (0, 1337), (1175, 948), (1620, 953), (8, 1711)], dtype=np.int32)
    lane_threshold = 1119
    x1, x2 = 876, 2160  # Detection area

    # Define the positions for the text annotations on the image (scaled for high resolution)
    text_position_left_lane = (50, 100)
    text_position_right_lane = (int(frame_width * 0.6), 100)  # Positioned based on video width
    intensity_position_left_lane = (50, 200)
    intensity_position_right_lane = (int(frame_width * 0.6), 200)

    # Define font, scale, and colors for the annotations (scaled for high resolution)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0  # Larger for high resolution video
    font_color = (255, 255, 255)    # White color for text
    background_color = (0, 0, 255)  # Red background for text

    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if not args.webcam else 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Define the codec and create VideoWriter object (only for video files)
    out = None
    if not args.webcam:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Output will be saved as: {args.output}")

    # Get total frame count for progress tracking (only for video files)
    total_frames = 0
    if not args.webcam:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames in video: {total_frames}")
    
    print("Processing... Press 'q' to quit, 'p' to pause/unpause, 's' to step frame by frame")
    
    frame_count = 0
    paused = False
    
    # Read until video is completed or user quits
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            if args.webcam:
                print("Failed to capture from webcam")
                break
            else:
                print("Video processing completed")
                break
        
        frame_count += 1
        
        # Print progress for video files
        if not args.webcam:
            if frame_count % 30 == 0:  # Print progress every 30 frames
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
        else:
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}...")
        
        # Create a copy of the original frame to modify
        detection_frame = frame.copy()
    
        # Black out the regions outside the specified vertical range (focus on main road)
        detection_frame[:x1, :] = 0  # Black out from top to x1
        detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame
        
        # Perform inference on the modified frame
        try:
            results = best_model.predict(detection_frame, imgsz=640, conf=args.conf, verbose=False)
            processed_frame = results[0].plot(line_width=2)
        except Exception as e:
            print(f"Error during inference: {e}")
            processed_frame = frame.copy()
        
        # Restore the original top and bottom parts of the frame
        processed_frame[:x1, :] = frame[:x1, :].copy()
        processed_frame[x2:, :] = frame[x2:, :].copy()        
        
        # Draw the quadrilaterals on the processed frame (lane boundaries)
        cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=4)  # Green for lane 1
        cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=4)  # Blue for lane 2
        
        # Draw detection area boundaries
        cv2.line(processed_frame, (0, x1), (frame_width, x1), (255, 255, 0), 3)  # Top boundary (yellow)
        cv2.line(processed_frame, (0, x2), (frame_width, x2), (255, 255, 0), 3)  # Bottom boundary (yellow)
        
        # Draw a center line for reference
        cv2.line(processed_frame, (lane_threshold, 0), (lane_threshold, frame_height), (255, 255, 255), 2)  # White center line
        
        # Add labels for lanes with better positioning
        label_y = int((x1 + x2) / 2)
        cv2.putText(processed_frame, "LANE 1", (200, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(processed_frame, "LANE 2", (50, int(frame_height * 0.8)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        
        # Retrieve the bounding boxes from the results
        if results and len(results) > 0 and results[0].boxes is not None:
            bounding_boxes = results[0].boxes

            # Initialize counters for vehicles in each lane
            vehicles_in_left_lane = 0
            vehicles_in_right_lane = 0

            # Loop through each bounding box to count vehicles in each lane
            for box in bounding_boxes.xyxy:
                # Get the center point of the bounding box
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                
                # Only count vehicles in the main detection area
                if x1 <= center_y <= x2:
                    # Use point-in-polygon test for more accurate lane detection
                    point = (center_x, center_y)
                    
                    # Check if point is in vertices1 (first lane)
                    if cv2.pointPolygonTest(vertices1, point, False) >= 0:
                        vehicles_in_left_lane += 1
                        # Draw a small circle to show detected vehicle center
                        cv2.circle(processed_frame, point, 8, (0, 255, 0), -1)
                    
                    # Check if point is in vertices2 (second lane)  
                    elif cv2.pointPolygonTest(vertices2, point, False) >= 0:
                        vehicles_in_right_lane += 1
                        # Draw a small circle to show detected vehicle center
                        cv2.circle(processed_frame, point, 8, (255, 0, 0), -1)
                    
                    # If not in either polygon, use fallback lane threshold method
                    elif center_x < lane_threshold:
                        vehicles_in_left_lane += 1
                        cv2.circle(processed_frame, point, 8, (0, 255, 255), -1)  # Yellow for fallback
                    else:
                        vehicles_in_right_lane += 1
                        cv2.circle(processed_frame, point, 8, (0, 255, 255), -1)  # Yellow for fallback
        else:
            vehicles_in_left_lane = 0
            vehicles_in_right_lane = 0
                
        # Determine the traffic intensity for each lane
        traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
        traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

        # Add text annotations with backgrounds (scaled for high resolution)
        # Lane 1 vehicle count
        cv2.rectangle(processed_frame, (text_position_left_lane[0]-20, text_position_left_lane[1] - 50), 
                      (text_position_left_lane[0] + 800, text_position_left_lane[1] + 20), background_color, -1)
        cv2.putText(processed_frame, f'Vehicles in Lane 1: {vehicles_in_left_lane}', text_position_left_lane, 
                    font, font_scale, font_color, 4, cv2.LINE_AA)

        # Lane 1 traffic intensity
        cv2.rectangle(processed_frame, (intensity_position_left_lane[0]-20, intensity_position_left_lane[1] - 50), 
                      (intensity_position_left_lane[0] + 600, intensity_position_left_lane[1] + 20), background_color, -1)
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_left_lane, 
                    font, font_scale, font_color, 4, cv2.LINE_AA)

        # Lane 2 vehicle count
        cv2.rectangle(processed_frame, (text_position_right_lane[0]-20, text_position_right_lane[1] - 50), 
                      (text_position_right_lane[0] + 800, text_position_right_lane[1] + 20), background_color, -1)
        cv2.putText(processed_frame, f'Vehicles in Lane 2: {vehicles_in_right_lane}', text_position_right_lane, 
                    font, font_scale, font_color, 4, cv2.LINE_AA)

        # Lane 2 traffic intensity
        cv2.rectangle(processed_frame, (intensity_position_right_lane[0]-20, intensity_position_right_lane[1] - 50), 
                      (intensity_position_right_lane[0] + 600, intensity_position_right_lane[1] + 20), background_color, -1)
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane, 
                    font, font_scale, font_color, 4, cv2.LINE_AA)

        # Add frame counter and progress info (top right corner)
        info_text = f"Frame: {frame_count}"
        if not args.webcam and total_frames > 0:
            progress = (frame_count / total_frames) * 100
            info_text += f" | Progress: {progress:.1f}%"
        
        # Add control instructions
        control_text = "Controls: Q=Quit, P=Pause, S=Step"
        
        # Position info text at top right
        info_x = frame_width - 600
        cv2.rectangle(processed_frame, (info_x-20, 20), (frame_width-20, 120), (0, 0, 0), -1)
        cv2.putText(processed_frame, info_text, (info_x, 60), font, 1.2, (255, 255, 255), 2)
        cv2.putText(processed_frame, control_text, (info_x, 100), font, 1.0, (255, 255, 255), 2)

        # Scale frame for display if needed
        if display_scale != 1.0:
            display_frame = cv2.resize(processed_frame, (display_width, display_height))
        else:
            display_frame = processed_frame
        
        # Display the processed frame with proper window controls
        cv2.namedWindow('Real-time Traffic Analysis - Indian Traffic', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Real-time Traffic Analysis - Indian Traffic', display_width, display_height)
        cv2.imshow('Real-time Traffic Analysis - Indian Traffic', display_frame)
        
        # Write frame to output video (only for video files)
        if out is not None:
            out.write(processed_frame)

        # Press Q on keyboard to exit the loop, P to pause, S to step
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('s') and paused:
            # Step one frame when paused
            pass
        elif paused:
            # If paused, wait for next keypress
            continue

    # Release everything
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    if not args.webcam:
        print(f"Processing complete! Output saved as: {args.output}")
        print(f"Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()
