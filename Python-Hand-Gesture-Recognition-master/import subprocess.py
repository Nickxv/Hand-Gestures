import subprocess
import time

# Define the touch coordinates (change these values as needed)
TOUCH_X = 500
TOUCH_Y = 500

# Simulate a touch tap
subprocess.call(["xdotool", "mousemove", str(TOUCH_X), str(TOUCH_Y)])
subprocess.call(["xdotool", "click", "1"])

# Simulate a touch swipe (change the end coordinates as needed)
subprocess.call(["xdotool", "mousemove", str(TOUCH_X), str(TOUCH_Y)])
subprocess.call(["xdotool", "mousedown", "1"])
subprocess.call(["xdotool", "mousemove", str(TOUCH_X + 100), str(TOUCH_Y)])
subprocess.call(["xdotool", "mouseup", "1"])

# Add more touch events as needed
