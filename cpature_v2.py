import cv2
import mediapipe as mp
import csv
import os
import time
import math
import numpy as np

# Configuration
MODEL_PATH = 'hand_landmarker.task'
CSV_FILE   = 'gesture_data_v2.csv'

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# Helpers
def calc_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def extract_features(landmarks):
    """ Extract the 10 key distances for gesture recognition """
    # Scale based on hand size
    scale = calc_distance(landmarks[0], landmarks[9]) or 1.0
    
    # Feature pairs (same as server.py/train_v3.py)
    pairs = [
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),  # Wrist to Fingertips
        (4, 8), (4, 12), (4, 16), (4, 20),          # Thumb to Fingertips
        (8, 12)                                     # Index to Middle
    ]
    return [calc_distance(landmarks[p1], landmarks[p2]) / scale for p1, p2 in pairs]

def main():
    # Initialize counts
    counts = {0: 0, 1: 0, 2: 0}

    # Check if file exists and read counts
    if os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader, None) # Skip header
                for row in reader:
                    if row:
                        try:
                            label = int(row[0])
                            if label in counts: counts[label] += 1
                        except ValueError: pass
            print(f"Loaded existing data: {counts}")
        except Exception as e:
            print(f"Error reading counts: {e}")
    else:
        # Create new
        with open(CSV_FILE, 'w', newline='') as f:
            header = ['label'] + [f'dist_{i}' for i in range(10)]
            csv.writer(f).writerow(header)
            print(f"Created new data file: {CSV_FILE}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    print("--- GESTURE RECORDER ---")
    print("[Q] Hover | [W] Draw | [E] Erase")
    print("Press ESC to exit.")
    
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            success, frame = cap.read()
            if not success: break
            
            # MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp)
            
            status = "Waiting..."
            
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                features = extract_features(lm)

                # Visualize the critical pinch distance (Thumb-Index)
                h, w, _ = frame.shape
                p1 = (int(lm[4].x * w), int(lm[4].y * h))
                p2 = (int(lm[8].x * w), int(lm[8].y * h))
                cv2.line(frame, p1, p2, (0, 255, 255), 2)
                
                # Input Handling
                key = cv2.waitKey(1)
                label = -1
                
                if key == ord('q'): 
                    label = 0; status = "REC: HOVER"
                elif key == ord('w'): 
                    label = 1; status = "REC: DRAW"
                elif key == ord('e'): 
                    label = 2; status = "REC: ERASE"
                
                # Save Data
                if label != -1:
                    with open(CSV_FILE, 'a', newline='') as f:
                        csv.writer(f).writerow([label] + features)
                    counts[label] += 1

            # UI Overlay
            cv2.rectangle(frame, (0, 0), (220, 140), (0, 0, 0), -1)
            cv2.putText(frame, f"Hover: {counts[0]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Draw:  {counts[1]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(frame, f"Erase: {counts[2]}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, status, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            cv2.imshow("Data Recorder", frame)
            if cv2.waitKey(1) == 27: break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()