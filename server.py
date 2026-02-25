import time
import os
import threading
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

# Configuration
MODEL_PATH = "gesture_v2.pth"
TASK_PATH  = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Sensitivity Settings
DEAD_ZONE_X = 0.02   # Side margins
DEAD_ZONE_Y = 0.08   # Bottom margin
SMOOTHING   = 0.6    # Default cursor smoothing
CONFIDENCE  = 0.85   # How sure the AI needs to be

# AI Model
class HandDistConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(2)
        self.fc1   = nn.Linear(320, 64)
        self.fc2   = nn.Linear(64, 3) 

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Network
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HandDistConv().to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[{device}] Model loaded successfully.")
except Exception as e:
    print(f"Warning: Could not load model '{MODEL_PATH}'. Error: {e}")

# Utils
class CursorStabilizer:
    """ Smooths cursor movement to reduce jitter """
    def __init__(self, min_speed=0.05, max_speed=0.7, drag=15.0):
        self.x, self.y = 0, 0
        self.min_s, self.max_s, self.drag = min_speed, max_speed, drag

    def update(self, target_x, target_y):
        if self.x == 0: 
            self.x, self.y = target_x, target_y
            
        dist = math.hypot(target_x - self.x, target_y - self.y)
        
        # Dynamic smoothing: Move fast = less smoothing
        factor = 1.0 / (1.0 + self.drag * dist)
        alpha = self.min_s + (self.max_s - self.min_s) * factor
        
        self.x = self.x * alpha + target_x * (1 - alpha)
        self.y = self.y * alpha + target_y * (1 - alpha)
        return self.x, self.y

def calc_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def extract_features(landmarks):
    """ Converts hand landmarks into the 10 point distance vector for the AI """
    # Scale based on hand size (wrist to middle finger knuckle)
    scale = calc_distance(landmarks[0], landmarks[9]) or 1.0
    
    # Key points to measure
    pairs = [
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),  # Wrist to fingertips
        (4, 8), (4, 12), (4, 16), (4, 20),          # Thumb to fingertips
        (8, 12)                                     # Index to Middle (spread)
    ]
    
    return [calc_distance(landmarks[p1], landmarks[p2]) / scale for p1, p2 in pairs], scale

# Main App
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Global State
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FPS, 60)
latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
frame_lock = threading.Lock()
current_gesture = 0  # 0=None, 1=Pinch, 2=Palm

def camera_loop():
    """ Continuously grabs frames from webcam """
    global latest_frame
    while True:
        success, frame = camera.read()
        if success:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            time.sleep(0.01)

def processing_loop():
    """ Main AI Loop: Finds hands -> Predicts Gesture -> Sends to UI """
    global current_gesture
    stabilizer = CursorStabilizer()
    
    # Setup MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=TASK_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            # 1. Get Frame
            with frame_lock: 
                frame = latest_frame.copy()
            
            # 2. Find Hands
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                
                # Raw Cursor Position (Midpoint of Thumb & Index)
                raw_x = (lm[4].x + lm[8].x) / 2
                raw_y = (lm[4].y + lm[8].y) / 2
                
                # Detect Fist
                is_fist = True
                wrist = lm[0]
                for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]:
                    if calc_distance(wrist, lm[tip]) > calc_distance(wrist, lm[mcp]):
                        is_fist = False
                        break

                # Check Dead Zones (Edges of screen)
                if raw_y > (1 - DEAD_ZONE_Y) or raw_x < DEAD_ZONE_X or raw_x > (1 - DEAD_ZONE_X):
                    current_gesture = 0
                    smooth_x, smooth_y = stabilizer.update(raw_x, raw_y)
                    socketio.emit("hand_data", {"x": smooth_x, "y": smooth_y, "pinch": False, "palm": False, "fist": is_fist})
                    time.sleep(0.01)
                    continue

                # 3. AI Prediction
                features, scale = extract_features(lm)
                
                # Convert to tensor and run model
                input_tensor = torch.tensor([features], dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs  = F.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                
                # Update state with hysteresis (debouncing)
                if conf.item() > CONFIDENCE:
                    current_gesture = pred.item()
                elif conf.item() < 0.4:
                    current_gesture = 0
                    
                # Sanity Check: If thumb/index far apart, force open
                if current_gesture == 1 and (calc_distance(lm[4], lm[8]) / scale) > 0.3:
                    current_gesture = 0

                # 4. Send Data
                smooth_x, smooth_y = stabilizer.update(raw_x, raw_y)
                socketio.emit("hand_data", {
                    "x": smooth_x, 
                    "y": smooth_y, 
                    "pinch": (current_gesture == 1), 
                    "palm": (current_gesture == 2),
                    "fist": is_fist
                })
            
            time.sleep(0.01)

def generate_mjpeg():
    """ Stream video to browser """
    while True:
        with frame_lock:
            frame = latest_frame.copy()
        
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.03)

# Server Routes
@app.route("/")
def index(): 
    return render_template("index.html")

@app.route("/video_feed")
def video_feed(): 
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=processing_loop, daemon=True).start()
    socketio.run(app, port=5000, debug=False, allow_unsafe_werkzeug=True)