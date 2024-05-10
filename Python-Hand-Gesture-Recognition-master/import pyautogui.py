import pyautogui
import time

# Define the screen resolution
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Define the touch coordinates (change these values as needed)
TOUCH_X = SCREEN_WIDTH // 2
TOUCH_Y = SCREEN_HEIGHT // 2

# Simulate a touch tap
pyautogui.click(x=TOUCH_X, y=TOUCH_Y)

# Simulate a touch swipe (change the end coordinates as needed)
pyautogui.moveTo(TOUCH_X, TOUCH_Y)
pyautogui.dragTo(TOUCH_X + 100, TOUCH_Y, duration=0.5)  # Swipe right

# Add more touch events as needed

