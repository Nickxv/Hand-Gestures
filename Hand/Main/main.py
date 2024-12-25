import cv2
import numpy as np
import pyautogui
import time
import mediapipe as mp

# Constants
FRAME_HEIGHT = 480
FRAME_WIDTH = 640
REGION_TOP = 0
REGION_BOTTOM = int(2 * FRAME_HEIGHT / 3)
REGION_LEFT = int(FRAME_WIDTH / 2)
REGION_RIGHT = FRAME_WIDTH
CALIBRATION_TIME = 30
BG_WEIGHT = 0.5
OBJ_THRESHOLD = 18
CURSOR_SPEED = 5

class HandData:
    def __init__(self, top, bottom, left, right, center_x):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.center_x = center_x
        self.prev_center_x = 0
        self.is_in_frame = False
        self.is_waving = False
        self.fingers = None
        self.gesture = None
        self.gesture_list = []
        self.last_gesture_time = 0
        self.finger_tap_start = False
        self.previous_fingers = None

    def update(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right

    def check_for_waving(self, center_x):
        self.prev_center_x = self.center_x
        self.center_x = center_x
        self.is_waving = abs(self.center_x - self.prev_center_x) > 3

def get_region(frame):
    region = frame[REGION_TOP:REGION_BOTTOM, REGION_LEFT:REGION_RIGHT]
    region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    region = cv2.GaussianBlur(region, (5, 5), 0)
    return region

def get_average(region):
    global background
    if background is None:
        background = region.copy().astype("float")
        return
    cv2.accumulateWeighted(region, background, BG_WEIGHT)

def segment(region):
    global hand
    if background is None:
          return None  # Handle case where background isn't initialized yet

    diff = cv2.absdiff(background.astype(np.uint8), region)
    thresholded_region = cv2.threshold(diff, OBJ_THRESHOLD, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresholded_region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        if hand is not None:
            hand.is_in_frame = False
        return None
    else:
        if hand is not None:
            hand.is_in_frame = True
        segmented_region = max(contours, key=cv2.contourArea)
        return thresholded_region, segmented_region

def get_hand_data(thresholded_image, segmented_image):
    global hand
    convex_hull = cv2.convexHull(segmented_image)
    top = tuple(convex_hull[convex_hull[:, :, 1].argmin()][0])
    bottom = tuple(convex_hull[convex_hull[:, :, 1].argmax()][0])
    left = tuple(convex_hull[convex_hull[:, :, 0].argmin()][0])
    right = tuple(convex_hull[convex_hull[:, :, 0].argmax()][0])
    center_x = int((left[0] + right[0]) / 2)
    if hand is None:
        hand = HandData(top, bottom, left, right, center_x)
    else:
        hand.update(top, bottom, left, right)
    if frames_elapsed % 6 == 0:
        hand.check_for_waving(center_x)
    current_fingers = count_fingers(thresholded_image)
    if hand.previous_fingers is not None and hand.previous_fingers != current_fingers:
        hand.gesture_list.append(current_fingers)
    hand.previous_fingers = current_fingers
    
    if frames_elapsed % 12 == 0:
        hand.fingers = most_frequent(hand.gesture_list)
        hand.gesture_list.clear()
        hand.gesture = hand.fingers

def count_fingers(thresholded_image):
    line_height = int(hand.top[1] + (0.2 * (hand.bottom[1] - hand.top[1])))
    line = np.zeros(thresholded_image.shape[:2], dtype=int)
    cv2.line(line, (thresholded_image.shape[1], line_height), (0, line_height), 255, 1)
    line = cv2.bitwise_and(thresholded_image, thresholded_image, mask=line.astype(np.uint8))
    contours, _ = cv2.findContours(line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fingers = 0
    for curr in contours:
        width = len(curr)
        if width < 3 * abs(hand.right[0] - hand.left[0]) / 4 and width > 5:
            fingers += 1
    return fingers

def most_frequent(input_list):
    if not input_list:
        return None
    counts = {}
    for item in reversed(input_list):
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get)

# Initialize cooldown variables
waving_cooldown_duration = 0.5
tap_cooldown_duration = 0.7

def perform_gesture_action(hand):
    current_time = time.time()
    if hand.is_waving:
        if current_time - hand.last_gesture_time > waving_cooldown_duration:
            pyautogui.click()
            hand.last_gesture_time = current_time
            print("Performed Wave Click")
    elif hand.fingers == 1 and hand.previous_fingers == 0:
         if current_time - hand.last_gesture_time > tap_cooldown_duration:
            pyautogui.click()
            hand.last_gesture_time = current_time
            print("Performed Single Finger Click")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

background = None
hand = None
frames_elapsed = 0

# Screen resolution
screen_width, screen_height = pyautogui.size()

# Video capture
cap = cv2.VideoCapture(1)

while True:
    success, image = cap.read()
    if not success:
        break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)

    image_rgb.flags.writeable = True
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
            tip_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            screen_x = int(tip_x * screen_width)
            screen_y = int(tip_y * screen_height)

            # Move the mouse
            pyautogui.moveTo(screen_x, screen_y)

             #region processing for click
            region = get_region(image)
            if frames_elapsed < CALIBRATION_TIME:
                get_average(region)
            else:
                 region_pair = segment(region)
                 if region_pair is not None:
                    thresholded_region, segmented_region = region_pair
                    get_hand_data(thresholded_region, segmented_region)
                    if hand.is_in_frame:
                         perform_gesture_action(hand)
    else:
        if hand is not None:
            hand.is_in_frame = False


    # Show the image
    cv2.imshow('MediaPipe Hands', image)
    frames_elapsed += 1

    # Break loop with q key
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()