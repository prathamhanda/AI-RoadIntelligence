AI-Powered Animal Detection System
Saving Lives with Real-Time Computer Vision

This project uses YOLOv8 and OpenCV to detect large animals on roadways — such as cows, horses, and elephants — while ignoring people and vehicles. The goal is to prevent traffic congestion and improve road safety through animal-aware video analysis.

🚀 Features
Detects only large road-blocking animals

Filters out humans and vehicles

Annotates frames visually (bounding boxes + labels)

Saves clean detection video as det_ele.mp4

Logs frame-by-frame detections to animal_log.csv

Works with manually input video (offline testing mode)

🧰 Dependencies
Make sure your Python environment has the following:

bash
ultralytics==x.x.x  # latest YOLOv8 package
opencv-python
torch
csv
You can install YOLOv8 with:

bash
pip install ultralytics
📂 Usage
Place your input video in the working folder and name it ele.mp4

Run the script:

bash
python elep.py
Your results:

Annotated output video: det_ele.mp4

Preview frame: preview_frame_filtered.jpg

Detection log: animal_log.csv

⚙️ Detection Logic
This script only visualizes and counts:

cow

horse

elephant (custom model required)

It ignores:

person

car

truck

all other classes in the YOLOv8 COCO dataset

🧠 Model Notes
Uses yolov8n.pt (COCO-trained) by default

For elephant detection, switch to a custom-trained model:

python
model = YOLO("yolov8_elephant.pt")
📈 Future Plans
Add live alert dashboard with Django

Train lightweight model for embedded devices (Raspberry Pi)

Visualize animal movement and heatmaps using Blender

Integrate GPS and time series prediction for smarter routing
