import cv2

# Load the pre-trained Haar Cascade classifier for hand detection
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')

# Function to detect hands in an image
def detect_hands(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect hands in the grayscale image
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected hands
    for (x, y, w, h) in hands:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return image

# Open a video capture object
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture object
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect hands in the frame
    frame_with_hands = detect_hands(frame)
    
    # Display the frame with detected hands
    cv2.imshow('Hand Detection', frame_with_hands)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
