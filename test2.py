import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# MediaPipe Setup
# -------------------------------
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

# -------------------------------
# Direction Function
# -------------------------------
def get_direction(wrist, index_tip):
    dx = index_tip[0] - wrist[0]
    dy = index_tip[1] - wrist[1]

    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "DOWN" if dy > 0 else "UP"

# -------------------------------
# Control Variables
# -------------------------------
cooldown = 1
last_action_time = 0
stable_frames = 0
last_direction = ""

# -------------------------------
# Start Camera
# -------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    direction_text = ""

    if result.hand_landmarks:
        h, w, _ = frame.shape
        hand_landmarks = result.hand_landmarks[0]

        points = []
        for lm in hand_landmarks:
            px = int(lm.x * w)
            py = int(lm.y * h)
            points.append((px, py))
            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        # Bounding Box
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        cv2.rectangle(frame,
                      (min(x_vals)-20, min(y_vals)-20),
                      (max(x_vals)+20, max(y_vals)+20),
                      (255, 0, 0), 2)

        wrist = points[0]
        index_tip = points[8]

        direction_text = get_direction(wrist, index_tip)

        # Stability check
        if direction_text == last_direction:
            stable_frames += 1
        else:
            stable_frames = 0

        last_direction = direction_text

        current_time = time.time()

        if stable_frames > 6 and current_time - last_action_time > cooldown:

            if direction_text == "RIGHT":
                keyboard.press_and_release("volume up")

            elif direction_text == "LEFT":
                keyboard.press_and_release("volume down")

            elif direction_text == "DOWN":
                keyboard.press_and_release("windows+d")

            last_action_time = current_time
            stable_frames = 0

    cv2.putText(frame, f"Direction: {direction_text}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0), 2)

    cv2.imshow("Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

