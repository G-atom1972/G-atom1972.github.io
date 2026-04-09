
import cv2
import mediapipe as mp
import numpy as np
import keyboard
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Ask Mode
# -------------------------------
mode = input("Enter mode (basic(1) / advanced(2)): ").strip().lower()


# -------------------------------
# MediaPipe Setup
# -------------------------------
base_options = python.BaseOptions(
    model_asset_path=r"location/hand_landmarker.task"
)

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
# BASIC MODE
# -------------------------------
if mode == "basic" or mode == 1 :

    cooldown = 1
    last_action_time = 0
    stable_frames = 0
    last_direction = ""

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

            wrist = points[0]
            index_tip = points[8]

            direction_text = get_direction(wrist, index_tip)

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

                elif direction_text == "UP":
                    keyboard.press_and_release("alt+tab")

                last_action_time = current_time
                stable_frames = 0

        cv2.putText(frame, f"Direction: {direction_text}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0), 2)

        cv2.imshow("Basic Mode", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------------------
# ADVANCED MODE (PINCH DRAW)
# -------------------------------

elif mode == "advanced" or mode == "2":

    cap = cv2.VideoCapture(0)
    canvas = None

    pinch_threshold = 40

    prev_x, prev_y = 0, 0  # for smooth drawing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        h, w, _ = frame.shape

        # Drawing box
        box_size = 640
        cx, cy = w // 2, h // 2

        x1 = cx - box_size // 2
        y1 = cy - box_size // 2
        x2 = cx + box_size // 2
        y2 = cy + box_size // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            # Index finger
            index_tip = hand_landmarks[8]
            ix = int(index_tip.x * w)
            iy = int(index_tip.y * h)

            # Thumb
            thumb_tip = hand_landmarks[4]
            tx = int(thumb_tip.x * w)
            ty = int(thumb_tip.y * h)

            # Draw tracking points
            cv2.circle(frame, (ix, iy), 8, (0, 255, 0), -1)
            cv2.circle(frame, (tx, ty), 8, (0, 0, 255), -1)

            distance = np.hypot(ix - tx, iy - ty)

            if distance < pinch_threshold:
                cv2.putText(frame, "PINCHING (DRAWING)",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0), 2)

                if x1 < ix < x2 and y1 < iy < y2:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = ix, iy

                    # ✨ SMOOTH LINE DRAWING
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy),
                             (255, 255, 255), 5)

                    prev_x, prev_y = ix, iy

            else:
                cv2.putText(frame, "OPEN HAND",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255), 2)

                # reset when not drawing
                prev_x, prev_y = 0, 0

        combined = cv2.add(frame, canvas)

        cv2.putText(combined, "ADVANCED MODE - SMOOTH DRAW",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255), 2)

        cv2.imshow("Gesture Paint", combined)

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break

        elif key & 0xFF == ord('c'):
            canvas = np.zeros_like(frame)

    cap.release()
    cv2.destroyAllWindows()
