🧠 SmartVision AI
Intelligent Multi-Class Object Recognition System (26 Classes)

SmartVision AI is an end-to-end Computer Vision & Deep Learning application that performs multi-class image classification and real-time multi-object detection using state-of-the-art CNN architectures and YOLO.
The system is trained on a curated 26-class subset of the COCO dataset and deployed as an interactive Streamlit web application.

🚀 Key Features

✅ 26-Class Image Classification using Transfer Learning

✅ Real-time Multi-Object Detection with YOLO

✅ Balanced COCO Subset (equal representation per class)

✅ Model Comparison Dashboard (Accuracy, Precision, Recall, F1)

✅ Interactive Streamlit Web App

✅ Deployment-ready Architecture

🏗️ System Architecture

User Flow:

Image Upload
   ↓
CNN Classification (EfficientNet-B0)
   ↓
YOLO Object Detection (Bounding Boxes + Confidence)
   ↓
Streamlit Visualization

🗂️ Dataset Details

Dataset: COCO 2017 (Common Objects in Context)

Source: Hugging Face – detection-datasets/coco

Total Images Used: 2,600 (≈100 images per class)

Classes: 26

Annotations: COCO JSON → YOLO format

Split: Train (70%) | Validation (15%) | Test (15%)

Why a Curated Subset?

Balanced classes (no class dominance)

Faster training & evaluation

Ideal for transfer learning

Suitable for real-world deployment demos

🏷️ Selected Object Classes (26)
🚗 Vehicles (7)

airplane, car, truck, bus, motorcycle, bicycle, train

👤 Human (1)

person

🚦 Outdoor Objects (3)

traffic light, stop sign, bench

🐾 Animals (6)

dog, cat, horse, bird, cow, elephant

🍽️ Kitchen & Food (5)

bottle, cup, bowl, pizza, cake

🪑 Furniture & Indoor (4)

chair, couch, bed, potted plant

🧠 Models Used
📷 Image Classification (Transfer Learning)
Model	Purpose
VGG16	Baseline CNN
ResNet18	Residual learning
MobileNet	Lightweight & fast
EfficientNet-B0 (Fully Unlocked)	Best performer

✔ Pretrained on ImageNet
✔ Fine-tuned on COCO subset
✔ Trained using PyTorch

🎯 Object Detection

Model: YOLOv8 (Ultralytics)

Type: Single-stage detector

Outputs: Bounding boxes, class labels, confidence scores

Supports: Multiple objects per image

📊 Model Performance Summary
🏆 Best Classification Model – EfficientNet-B0 (Fully Unlocked)
Metric	Value
Accuracy	81.65%
Precision	81.88%
Recall	81.90%
F1 Score	81.86%
Inference Time	~40–60 ms (CPU)
Model Size	41.49 MB

Performance measured on a balanced real-world dataset, making results more reliable than class-skewed benchmarks.

🎯 YOLOv8 Detection Performance

mAP@0.5: ~85–90%

Precision: ~97%

Recall: ~85%

Inference Speed: ~30–50 FPS (GPU)

Objects Detected: 1–10+ per image

🖥️ Streamlit Application Pages

Home – Project overview & instructions

Image Classification – Upload & classify images

Object Detection – YOLO bounding box visualization

Model Performance – Metrics & confusion matrices

About – Dataset, models & tech stack

🛠️ Technology Stack

Language: Python

Deep Learning: PyTorch

Computer Vision: OpenCV

Object Detection: YOLOv8 (Ultralytics)

Web App: Streamlit

Deployment: Streamlit Cloud / Hugging Face Spaces

📦 Project Structure
SmartVision-AI/
│
├── Streamlit Cloud/
│   ├── pages/
│   │   ├── classification.py
│   │   ├── object_detection.py
│   │   ├── model_performance.py
│   │   └── about.py
│
├── models/
│   ├── best_efficientnetb0_smartvision_unlocked.pth
│
├── images/
│   ├── YOLO/
│   ├── EfficientNet/
│   ├── MobileNet/
│   ├── ResNet18/
│   └── VGG16/
│
├── requirements.txt
└── README.md

▶️ How to Run Locally
# Clone repository
git clone https://github.com/your-username/SmartVision-AI.git
cd SmartVision-AI

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run Streamlit\ Cloud/app.py

🎯 Business Use Cases

Smart Cities & Traffic Monitoring

Retail & Visual Search

Security & Surveillance

Wildlife Conservation

Healthcare Monitoring

Smart Homes & IoT

Agriculture & Livestock Monitoring

Logistics & Warehousing

📌 Project Highlights

✔ Balanced real-world dataset
✔ End-to-end ML pipeline
✔ Model comparison & evaluation
✔ Production-ready deployment
✔ Clean modular Streamlit design

👨‍💻 Developer

Project: SmartVision AI
Domain: Computer Vision & Artificial Intelligence
Type: Capstone / Final Project

Built following industry best practices in deep learning, evaluation, and deployment.

📜 License

This project is for educational and demonstration purposes.