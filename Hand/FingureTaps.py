import cv2
import numpy as np
import pyautogui
import time

# Load the cascade
hand_cascade = cv2.CascadeClassifier('hg.xml')

# To capture video from webcam.
cap = cv2.VideoCapture(1)

# To store the previous hand center.
prev_center = None

# To store the previous mouse position.
prev_mouse_pos = None

# To store the finger tap count.
finger_tap_count = 0

while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect handsq
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around the hands
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Find the center of the hand
        center = (x + w // 2, y + h // 2)

        # Check if the hand is close to the previous hand
        if prev_center is not None and np.linalg.norm(np.array(center) - np.array(prev_center)) < 50:
            # Simulate a left mouse click
            pyautogui.click()

            # Increment the finger tap count
            finger_tap_count += 1

            # Print the finger tap count
            print(f"Finger tap count: {finger_tap_count}")

            # Wait for a second before allowing another click
            time.sleep(1)

        # Update the previous hand center
        prev_center = center

        # Draw the center of the hand
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Gesture Control', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
