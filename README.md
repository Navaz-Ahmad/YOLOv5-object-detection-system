

# 🔍 YOLOv5 Object Detection Web App

This is a Flask-based web application that uses **YOLOv5** to perform real-time object detection on:

* 📷 Uploaded images
* 🎥 Uploaded videos
* 🎦 Live camera feed

The app detects and highlights objects using YOLOv5 and displays results directly in your browser.


## 📁 Features

* 🖼 Upload and detect objects in **images**
* 📹 Upload and detect objects in **videos**
* 📷 Start **live camera detection**
* 💻 Web-based and easy-to-use interface
* 🎯 Built on YOLOv5 with high accuracy and speed

---

## 📸 Demo Preview

> Coming soon...

---

## 🧠 Technology Stack

| Component         | Tech Used                |
| ----------------- | ------------------------ |
| Backend Framework | Flask (Python)           |
| Object Detection  | YOLOv5 (Ultralytics)     |
| Frontend          | HTML5 + CSS3             |
| Live Feed         | OpenCV with MJPEG Stream |
| Deployment        | Localhost or Cloud       |

---

## 🛠️ Installation and Setup

### 1. Clone this repository

```bash
git clone https://github.com/your-username/yolov5-object-detection-app.git
cd yolov5-object-detection-app
```

### 2. Install dependencies

Make sure you are using **Python 3.7+**. Create a virtual environment and install:

```bash
pip install -r requirements.txt
```

### 3. Folder Structure

```bash
.
├── app.py                       # Main Flask app
├── detect_image.py             # Image detection logic
├── detect_video.py             # Video detection logic
├── detect_camera.py            # Camera detection logic
├── templates/
│   ├── index.html              # Main webpage
│   ├── image_result.html       # Image results
│   ├── video_result.html       # Video results
│   └── camera.html             # Live feed
├── static/
│   ├── uploads/                # Uploaded media
│   └── results/                # Output detections
└── runs/                       # YOLOv5 output
```

### 4. Run the app

```bash
python app.py
```

Visit: [http://localhost:5000](http://localhost:5000)

---

## 💡 How It Works

* User uploads an image/video or starts the camera.
* Flask passes the file to the detection module.
* YOLOv5 detects objects and saves the result.
* The output is shown back in the browser.

---

## 📂 Upload & Result Paths

* Uploads: `static/uploads/`
* Results: `static/results/`

---

## 🙋‍♂️ Author

**Syed Navaz Ahmad**


## ⭐ Acknowledgements

* [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
* Flask framework
* OpenCV


