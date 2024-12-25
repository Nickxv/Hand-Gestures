import cv2
import numpy as np

# Load pre-trained hand detection model
hand_cascade = cv2.CascadeClassifier(r'\C:\Users\nlada\Desktop\Python\Hand\hg.xml') # Or any other pre-trained model

# Function to detect hands and draw rectangles around them
def detect_hands(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in hands:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image, hands

# Function to recognize gestures
def recognize_gesture(hand):
    # Add your gesture recognition logic here
    # This could involve contour analysis, deep learning models, etc.
    pass

# Main function
def main():
    cap = cv2.VideoCapture(1)  # Use 0 for webcam
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect hands
        frame_with_hands, hands = detect_hands(frame)
        
        # Gesture recognition
        for hand in hands:
            recognize_gesture(hand)
        
        # Display the frame
        cv2.imshow('Hand Detection', frame_with_hands)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
