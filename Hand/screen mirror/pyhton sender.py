import os
import subprocess
import cv2
import numpy as np

# Install scrcpy and its dependencies
os.system("sudo snap install scrcpy")
os.system("sudo dnf install SDL2 android-tools")  # For Fedora or CentOS
os.system("sudo apt install SDL2 android-tools-adb")  # For Ubuntu or Debian
os.system("python3 -m pip install -r requirements.txt --user")

# Enable Developer Mode on your Android phone
# Go to Settings > About phone > Developer Mode

# Connect your phone to your PC using a USB cable
# Run the following command to start scrcpy
scrcpy_process = subprocess.Popen(['scrcpy'], stdout=subprocess.PIPE)

while True:
    # Read the screen image from scrcpy
    img = cv2.imdecode(np.frombuffer(scrcpy_process.stdout.read(921600), np.uint8), cv2.IMREAD_COLOR)

    # Display the screen image
    cv2.imshow('Screen', img)
    cv2.waitKey(1)
    
scrcpy_process = subprocess.Popen(['scrcpy', '--max-size', '720'], stdout=subprocess.PIPE)