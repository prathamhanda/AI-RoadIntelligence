from ultralytics import YOLO
import cv2
import csv

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Cow and horse detection (COCO-trained)
model.to('cpu')  # Or 'cuda' if you have a GPU

# Video input/output
video_path = "cow2.mp4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("det_ele.mp4", fourcc, fps, (width, height))

# Target animals
target_animals = ["cow", "horse"]

# CSV logging
csv_file = open("animal_log.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Animal", "Count"])

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Inference with no verbose prints
    results = model(frame, conf=0.25, verbose=False)

    # Custom annotation to exclude unwanted classes
    annotated_frame = frame.copy()
    animal_counts = {animal: 0 for animal in target_animals}

    if results[0].boxes:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label in target_animals:
                animal_counts[label] += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    for animal, count in animal_counts.items():
        if count > 0:
            print(f"🛑 Frame {frame_count}: {count} {animal}(s) detected")
            csv_writer.writerow([frame_count, animal, count])

    if frame_count == 1:
        cv2.imwrite("preview_frame_filtered.jpg", annotated_frame)

    out.write(annotated_frame)

cap.release()
out.release()
csv_file.close()

print("✅ Detection done. Output saved as det_ele.mp4")
