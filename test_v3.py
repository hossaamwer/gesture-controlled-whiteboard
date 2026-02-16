import time
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mediapipe as mp

# Configuration
MODEL_PATH = "gesture_v2.pth"
TASK_PATH  = "hand_landmarker.task"
CLASSES    = ["HOVER", "DRAW", "ERASE"]
COLORS     = [(0, 255, 0), (0, 165, 255), (0, 0, 255)] # Green, Orange, Red

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

# Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = HandDistConv().to(device)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"[{device}] V2 Model loaded successfully.")
except Exception as e:
    print(f"Error: Could not load '{MODEL_PATH}'. Run train_v2.py first.")
    print(f"Details: {e}")
    exit()

# Helpers
def calc_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def extract_features(landmarks):
    # Scale based on hand size
    scale = calc_distance(landmarks[0], landmarks[9]) or 1.0
    
    # 10 Key distances
    pairs = [
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
        (4, 8), (4, 12), (4, 16), (4, 20),
        (8, 12)
    ]
    return [calc_distance(landmarks[p1], landmarks[p2]) / scale for p1, p2 in pairs]

# Main Loop
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # MediaPipe Setup
    BaseOptions = mp.tasks.BaseOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=TASK_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1
    )
    
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        print("Starting camera... Press ESC to exit.")
        
        while True:
            success, frame = cap.read()
            if not success: break
            
            # Convert for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(mp_image, int(time.time() * 1000))
            
            # Draw UI background
            cv2.rectangle(frame, (0, 0), (220, 160), (0, 0, 0), -1)
            
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                
                # Predict
                features = extract_features(lm)
                input_tensor = torch.tensor([features], dtype=torch.float32).to(device)
                
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs  = F.softmax(logits, dim=1)
                    conf, idx = torch.max(probs, dim=1)
                    
                # Visualize
                for i, label in enumerate(CLASSES):
                    score = probs[0][i].item()
                    bar_width = int(score * 180)
                    color = COLORS[i]
                    
                    # Highlight active gesture
                    if i == idx.item():
                        cv2.putText(frame, f"{label} ({int(score*100)}%)", (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Draw bars
                    y = 70 + i * 35
                    cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    cv2.rectangle(frame, (80, y-10), (80 + bar_width, y+5), color, -1)
            
            cv2.imshow("Gesture V2 Test", frame)
            if cv2.waitKey(1) == 27: # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()