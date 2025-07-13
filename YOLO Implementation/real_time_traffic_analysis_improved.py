import cv2
import numpy as np
import argparse
import os
from ultralytics import YOLO

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Vehicle Detection and Traffic Analysis')
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
    heavy_traffic_threshold = 10

    # Define the vertices for the quadrilaterals (lane boundaries)
    vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
    vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

    # Define the vertical range for the slice and lane threshold
    x1, x2 = 325, 635 
    lane_threshold = 609

    # Define the positions for the text annotations on the image
    text_position_left_lane = (10, 50)
    text_position_right_lane = (820, 50)
    intensity_position_left_lane = (10, 100)
    intensity_position_right_lane = (820, 100)

    # Define font, scale, and colors for the annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)    # White color for text
    background_color = (0, 0, 255)  # Red background for text
    
    # Initialize video capture
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

    # Get video properties for output
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if not args.webcam else 20
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object (only for video files)
    out = None
    if not args.webcam:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Output will be saved as: {args.output}")

    print("Processing... Press 'q' to quit")
    
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
        
        # Create a copy of the original frame to modify
        detection_frame = frame.copy()
    
        # Black out the regions outside the specified vertical range
        detection_frame[:x1, :] = 0  # Black out from top to x1
        detection_frame[x2:, :] = 0  # Black out from x2 to the bottom of the frame
        
        # Perform inference on the modified frame
        try:
            results = best_model.predict(detection_frame, imgsz=640, conf=args.conf, verbose=False)
            processed_frame = results[0].plot(line_width=1)
        except Exception as e:
            print(f"Error during inference: {e}")
            processed_frame = frame.copy()
        
        # Restore the original top and bottom parts of the frame
        processed_frame[:x1, :] = frame[:x1, :].copy()
        processed_frame[x2:, :] = frame[x2:, :].copy()        
        
        # Draw the quadrilaterals on the processed frame
        cv2.polylines(processed_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(processed_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Retrieve the bounding boxes from the results
        if results and len(results) > 0 and results[0].boxes is not None:
            bounding_boxes = results[0].boxes

            # Initialize counters for vehicles in each lane
            vehicles_in_left_lane = 0
            vehicles_in_right_lane = 0

            # Loop through each bounding box to count vehicles in each lane
            for box in bounding_boxes.xyxy:
                # Check if the vehicle is in the left lane based on the x-coordinate of the bounding box
                if box[0] < lane_threshold:
                    vehicles_in_left_lane += 1
                else:
                    vehicles_in_right_lane += 1
        else:
            vehicles_in_left_lane = 0
            vehicles_in_right_lane = 0
                
        # Determine the traffic intensity for each lane
        traffic_intensity_left = "Heavy" if vehicles_in_left_lane > heavy_traffic_threshold else "Smooth"
        traffic_intensity_right = "Heavy" if vehicles_in_right_lane > heavy_traffic_threshold else "Smooth"

        # Add text annotations with backgrounds
        # Left lane vehicle count
        cv2.rectangle(processed_frame, (text_position_left_lane[0]-10, text_position_left_lane[1] - 25), 
                      (text_position_left_lane[0] + 460, text_position_left_lane[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Vehicles in Left Lane: {vehicles_in_left_lane}', text_position_left_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Left lane traffic intensity
        cv2.rectangle(processed_frame, (intensity_position_left_lane[0]-10, intensity_position_left_lane[1] - 25), 
                      (intensity_position_left_lane[0] + 460, intensity_position_left_lane[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_left}', intensity_position_left_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Right lane vehicle count
        cv2.rectangle(processed_frame, (text_position_right_lane[0]-10, text_position_right_lane[1] - 25), 
                      (text_position_right_lane[0] + 460, text_position_right_lane[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Vehicles in Right Lane: {vehicles_in_right_lane}', text_position_right_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Right lane traffic intensity
        cv2.rectangle(processed_frame, (intensity_position_right_lane[0]-10, intensity_position_right_lane[1] - 25), 
                      (intensity_position_right_lane[0] + 460, intensity_position_right_lane[1] + 10), background_color, -1)
        cv2.putText(processed_frame, f'Traffic Intensity: {traffic_intensity_right}', intensity_position_right_lane, 
                    font, font_scale, font_color, 2, cv2.LINE_AA)

        # Display the processed frame
        cv2.imshow('Real-time Traffic Analysis', processed_frame)
        
        # Write frame to output video (only for video files)
        if out is not None:
            out.write(processed_frame)

        # Press Q on keyboard to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    if not args.webcam:
        print(f"Processing complete! Output saved as: {args.output}")

if __name__ == "__main__":
    main()
