# Traffic_Light_YOLOv8
Real-time Traffic Light Detection and Classification using YOLOv8 for Deep Learning Course
# [cite_start]LBLM368: Real-Time Traffic Light Detection and Classification (YOLOv8/CNN) [cite: 5, 4]

## 📌 Project Status
Completed (Training Successful)

## 💡 Overview
[cite_start]The primary goal of this project was to develop a machine learning model using the modern Convolutional Neural Network (CNN) architecture, **YOLOv8**, to detect the location and classify the active state ('Red', 'Yellow', 'Green') of traffic lights in video streams[cite: 5, 18]. [cite_start]The model’s development aims to directly contribute to road safety by providing high accuracy and low latency for autonomous systems (ADAS)[cite: 19, 20].

---

## 2. 🚀 Setup and Execution
## 2.2. Training the Model 

To reproduce the training results, the dataset must be downloaded from the source (Roboflow) and placed in Google Drive.

The training command used was:

```bash
# Example command used in Colab
from ultralytics import YOLO

# Load pre-trained weights
model = YOLO('yolov8n.pt')

# Train the model for 50 epochs (data path must be set correctly)
model.train(data='/content/traffic-light/data.yaml', 
            epochs=50, 
            imgsz=640)

### 2.1. Prerequisites and Installation
[cite_start]The project was executed in a **Google Colab** environment [cite: 50] [cite_start]using **Python** [cite: 49] [cite_start]and the **Ultralytics YOLOv8** framework[cite: 48].

```bash
# Install the necessary library
!pip install ultralytics [cite: 71]
# Ensure best.pt and the video file are in the same directory.
# 'source' is the path to the video/image to be tested.
from ultralytics import YOLO
model = YOLO('runs/detect/train/weights/best.pt') [cite: 74, 137]
model.predict(source='test_video.mp4', save=True, conf=0.5) [cite: 68]
Metric,Value
Accuracy,96.5% 
mAP50 (Average Precision),0.975 
Precision,0.94 
Recall,0.95 
F1-Score,0.945
The custom Traffic Light Detection Dataset was sourced and adapted from Roboflow Universe.
