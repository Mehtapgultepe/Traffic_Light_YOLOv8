# LBLM368: Real-Time Traffic Light Detection and Classification (YOLOv8/CNN)

## 📌 Project Status
Completed (Training Successful)

## 💡 Overview
The primary goal of this project was to develop a machine learning model using the modern Convolutional Neural Network (CNN) architecture, **YOLOv8**, to detect the location and classify the active state ('Red', 'Yellow', 'Green') of traffic lights in video streams. The model’s development aims to directly contribute to road safety by providing high accuracy and low latency for autonomous systems (ADAS).

---

## 2. 🚀 Setup and Execution

### 2.1. Prerequisites and Installation
The project was executed in a **Google Colab** environment using **Python** and the **Ultralytics YOLOv8** framework.

```bash
# Install the necessary library
!pip install ultralytics
# 1. Mount Google Drive (Requires user permission)
from google.colab import drive
drive.mount('/content/drive')

# 2. Copy the zip file from Drive to the local Colab disk
# NOTE: 'traffic-light.zip' must be present in your Google Drive root!
!cp "/content/drive/MyDrive/traffic-light.zip" /content/

# 3. Unzip the data silently for use by the model
!unzip -q /content/traffic-light.zip
from ultralytics import YOLO

# Load pre-trained weights
model = YOLO('yolov8n.pt')

# Train the model for 50 epochs
model.train(data='/content/traffic-light/data.yaml', 
            epochs=50, 
            imgsz=640)
# Load the best trained model (path is auto-generated in /runs/detect/...)
model = YOLO('runs/detect/train/weights/best.pt')

# Predict on a local video source
# NOTE: The 'test_video.mp4' file must be uploaded to Colab's content directory.
model.predict(source='test_video.mp4', save=True, conf=0.5)
Harika! İşte, tüm adımları ve yapısal düzenlemeleri içeren, GitHub README.md dosyanıza doğrudan kopyalayıp yapıştırabileceğiniz nihai ve düzenlenmiş İngilizce içerik:

Markdown
# LBLM368: Real-Time Traffic Light Detection and Classification (YOLOv8/CNN)

## 📌 Project Status
Completed (Training Successful)

## 💡 Overview
The primary goal of this project was to develop a machine learning model using the modern Convolutional Neural Network (CNN) architecture, **YOLOv8**, to detect the location and classify the active state ('Red', 'Yellow', 'Green') of traffic lights in video streams. The model’s development aims to directly contribute to road safety by providing high accuracy and low latency for autonomous systems (ADAS).

---

## 2. 🚀 Setup and Execution

### 2.1. Prerequisites and Installation
The project was executed in a **Google Colab** environment using **Python** and the **Ultralytics YOLOv8** framework.

```bash
# Install the necessary library
!pip install ultralytics
2.2. Data Preparation and Loading (Crucial Step)

The dataset must be available as 'traffic-light.zip' in Google Drive. These commands copy and unzip the data into the Colab environment for training.

Python
# 1. Mount Google Drive (Requires user permission)
from google.colab import drive
drive.mount('/content/drive')

# 2. Copy the zip file from Drive to the local Colab disk
# NOTE: 'traffic-light.zip' must be present in your Google Drive root!
!cp "/content/drive/MyDrive/traffic-light.zip" /content/

# 3. Unzip the data silently for use by the model
!unzip -q /content/traffic-light.zip
2.3. Training the Model

This step trains the YOLOv8n model for 50 epochs using the extracted dataset.

Python
from ultralytics import YOLO

# Load pre-trained weights
model = YOLO('yolov8n.pt')

# Train the model for 50 epochs
model.train(data='/content/traffic-light/data.yaml', 
            epochs=50, 
            imgsz=640)
2.4. Running Inference (Testing)

After training, the final model (best.pt) is used to predict objects on a local test video.

Python
# Load the best trained model (path is auto-generated in /runs/detect/...)
model = YOLO('runs/detect/train/weights/best.pt')

# Predict on a local video source
# NOTE: The 'test_video.mp4' file must be uploaded to Colab's content directory.
model.predict(source='test_video.mp4', save=True, conf=0.5)
---
## $3. 📊 Key Performance Metrics
The model demonstrated strong performance on the validation set after 50 epochs:

Metric	Value
Accuracy	96.5%
mAP50 	0.975
Precision	0.94
Recall	0.95
F1-Score	0.945
---
## 4. 📝 Data Source
The custom Traffic Light Detection Dataset was sourced and adapted from Roboflow Universe.
---
