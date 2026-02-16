
# Gesture Controlled Whiteboard  
## AI-Powered Real-Time Gesture-Controlled Whiteboard System

Gesture Controlled Whiteboard is a computer vision–driven interactive drawing and productivity platform that transforms hand gestures into a fully contactless digital workspace.

By combining MediaPipe hand tracking, a custom PyTorch convolutional model, and a low-latency WebSocket architecture, the system enables smooth real-time drawing, PDF annotation, and multi-page project management without requiring physical input devices.

---

# Core Capabilities

## Intelligent Gesture Recognition

### Custom Deep Learning Model (PyTorch CNN)
A lightweight convolutional neural network is used to classify gesture states (e.g., Pinch vs. Palm) and improve robustness under lighting and motion variability.

### MediaPipe 3D Landmark Tracking
Tracks 21 hand landmarks in real time to achieve precise cursor positioning and gesture detection.

### Dynamic Cursor Stabilization Algorithm
A custom distance-based interpolation and deadzone filtering mechanism eliminates micro-jitter while preserving responsiveness during fast movements.

---

# Real-Time Rendering Engine

## Dual-Layer Ink Smoothing
- Server-side deadzone filtering to suppress sensor noise  
- Client-side curve interpolation (Bezier-based) for professional-grade digital ink  

## Multi-Layer Canvas Architecture
- Camera / Background layer  
- Document (PDF/Image) layer  
- Grid overlay layer  
- Drawing layer  

This separation prevents unnecessary redraw cycles and improves performance efficiency.

## Creative Toolkit
- Free-draw mode  
- Geometric shapes (Line, Rectangle, Circle, Triangle, Arrow)  
- Emoji placement (30+ icons)  
- Adjustable brush size  
- Variable eraser thickness  
- 13-color background engine  

---

# Workspace & Productivity System

## PDF & Image Integration
- Real-time PDF rendering via PDF.js  
- Multi-page document annotation  
- Responsive scaling for different screen sizes  

## Grid Engine
- Line and dot grid styles  
- Adjustable spacing  
- Designed for drafting and technical sketches  

## State Management
- Multi-page workspace  
- Undo stack  
- Project save/load as `.json`  
- High-resolution `.png` export  

---

# Technologies Used

## Backend
- Python  
- Flask  
- Flask-SocketIO (real-time WebSockets)  
- OpenCV  

## AI / ML
- PyTorch (Convolutional Neural Network)  
- MediaPipe (Hand Landmark Detection)  
- CUDA (optional GPU acceleration)  

## Frontend
- HTML5 Canvas API  
- JavaScript (ES6+)  
- PDF.js  

---

# System Architecture Overview

### 1. Vision Processing Layer (Python)
- Frame capture via OpenCV  
- Landmark extraction using MediaPipe  
- Gesture refinement using PyTorch CNN  
- Deadzone filtering for jitter suppression  

### 2. Real-Time Communication Layer
- Bidirectional WebSocket communication via Flask-SocketIO  
- Low-latency coordinate streaming  

### 3. Rendering & UI Layer (Frontend)
- Multi-layer canvas rendering  
- Curve-based stroke smoothing  
- Modular drawing tools  
- State persistence and export system  

---

# Performance Considerations

- Low-latency WebSocket communication  
- Lightweight CNN to maintain real-time inference  
- Layer-isolated rendering to avoid full-canvas redraw  
- Controlled undo stack to limit memory overhead  
- GPU acceleration support (CUDA) for faster inference  

---

# Engineering Evolution & Challenges

The current architecture of Gesture Controlled Whiteboard is the result of iterative problem-solving and multiple technical pivots to overcome real-world computer vision limitations.

## 1. The Pivot from Custom CNNs to Hybrid Tracking

**Initial Approach:**  
Attempted a fully custom PyTorch-based hand detection system trained on self-collected datasets.

**Challenge:**  
Severe overfitting due to limited dataset diversity and high sensitivity to lighting and background changes.

**Solution:**  
Pivoted to a hybrid architecture—leveraging MediaPipe for high-fidelity landmark detection while using a lightweight PyTorch CNN strictly for gesture classification. This significantly improved robustness and reduced training complexity.

---

## 2. Deep Dive into Hand Architecture & Spatial Logic

**The Challenge:**  
Distinguishing subtle gestures (e.g., Pinch vs. Point) required identifying the most informative landmark relationships.

**The Research:**  
Conducted extensive experimentation using Euclidean distance calculations between key landmarks (e.g., thumb tip vs. index tip) to define stable gesture thresholds.

**The Learning:**  
This required understanding hand anatomy and spatial geometry, leading to a robust filtering logic that isolates intentional gestures from accidental movements.

---

## 3. Moving Beyond the OpenCV GUI

**Challenge:**  
The native OpenCV GUI was visually limited and unsuitable for a modern creative tool.

**Solution:**  
Migrated to a Web-based architecture (Flask + JavaScript), enabling:
- A modern glassmorphism UI  
- Improved rendering flexibility  
- Modular frontend feature expansion  

---

## 4. Solving the "Jagged Line" Problem

**Challenge:**  
Initial coordinate streaming caused unstable, jagged lines due to sensor noise.

**Engineering Fixes:**

- **Data Streaming:** Broadcast raw landmark data via WebSockets for client-side rendering.  
- **Stabilization:** Implemented a custom cursor stabilizer and dual-layer smoothing algorithm (deadzone + Bezier interpolation) to achieve fluid, professional-grade digital ink.

---

## 5. Scaling the Feature Set

Once the core real-time engine stabilized, the architecture was expanded to support advanced productivity features:

- **Document Context:** Real-time PDF/Image annotation using PDF.js  
- **State Control:** Undo stack and multi-page management  
- **Technical Drafting:** Custom grid system and geometric shape engine  
 
---

## Project Structure

A breakdown of the repository organization and the purpose of each key file:

```text
gesture-controlled-whiteboard/
├── server.py              # The main entry point; hosts the Flask-SocketIO backend
├── requirements.txt       # List of all Python dependencies for the project
├── README.md              # Technical documentation and project overview
├── gesture_v2.pth         # The trained PyTorch CNN model for gesture classification
├── hand_landmarker.task   # MediaPipe's pre-trained hand tracking asset
├── gesture_data_v2.csv    # The captured dataset containing hand landmark coordinates
├── train_v2.py            # Script for training the PyTorch model on the captured data
├── test_v2.py             # Evaluation script to verify model accuracy
└── templates/             # Folder containing frontend assets
    └── index.html         # The UI; handles Canvas rendering and WebSocket communication
```

---

# Getting Started

## Installation

```bash
git clone https://github.com/hossaamwer/gesture-controlled-whiteboard.git
cd gesture-controlled-whiteboard
pip install -r requirements.txt
python server.py
```

Open your browser at:

```
http://localhost:5000
```

---

# Gesture Guide

**Pinch (Thumb + Index)**  
Draw, interact with UI elements, or place selected shapes/emojis.

**Open Palm**  
Activates eraser mode within the cursor radius.

**Pinch + Drag**  
Create geometric shapes; release to finalize.

---

# Project Objective

This project explores the integration of deep learning, computer vision, and real-time web rendering into a cohesive interactive system.

The primary focus was architectural clarity, latency optimization, and creating a smooth, intuitive contactless drawing experience.

---

# Future Enhancements

- Advanced gesture classification models  
- Model quantization for optimized inference  
- WebGL-based rendering pipeline  
- Stroke-vector undo system instead of image snapshots  
- Multi-user collaborative mode  
