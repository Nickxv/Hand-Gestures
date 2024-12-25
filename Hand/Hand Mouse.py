import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Function to calculate the distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Screen resolution
screen_width, screen_height = pyautogui.size()

# Video capture
cap = cv2.VideoCapture(0)

# Variable to keep track of previous state for detecting tap
previous_tap = False

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of the index finger tip and the palm (wrist)
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

            # Convert to screen coordinates for mouse movement
            tip_x = index_tip.x
            tip_y = index_tip.y
            screen_x = int(tip_x * screen_width)
            screen_y = int(tip_y * screen_height)

            # Move the mouse to the index finger tip position
            pyautogui.moveTo(screen_x, screen_y)

            # Get the positions in normalized coordinates
            index_tip_x, index_tip_y = index_tip.x, index_tip.y
            index_dip_x, index_dip_y = index_dip.x, index_dip.y
            wrist_x, wrist_y = wrist.x, wrist.y

            # Calculate the distance between the index tip and the wrist (or DIP joint)
            distance_tip_wrist = calculate_distance(index_tip_x, index_tip_y, wrist_x, wrist_y)
            distance_tip_dip = calculate_distance(index_tip_x, index_tip_y, index_dip_x, index_dip_y)

            # Detect tap gesture when the index tip moves close to the DIP (like a tapping motion)
            if distance_tip_dip < 0.04 and not previous_tap:
                pyautogui.click()
                previous_tap = True
            elif distance_tip_dip >= 0.04:
                previous_tap = False

    # Show the image
    cv2.imshow('MediaPipe Hands', image)

    # Break loop with q key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
