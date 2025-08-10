import os
import cv2

# Check if files exist
print("yolov8m.pt exists:", os.path.exists("yolov8m.pt"))
print("yolov4.weights exists:", os.path.exists("yolov4.weights"))
print("coco.names exists:", os.path.exists("coco.names"))

# Try reading the config file
try:
    with open("yolov8m.pt", "r") as f:
        print("First line of yolov8m.pt:", f.readline())
except Exception as e:
    print("Error reading yolov8m.pt:", e)

# Try loading the network
try:
    net = cv2.dnn.readNetFromDarknet("yolov8m.pt", "yolov4.weights")
    print("Network loaded successfully!")
except Exception as e:
    print("Error loading network:", e)
