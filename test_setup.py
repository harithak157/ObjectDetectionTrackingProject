import cv2
import numpy as np
import os

print("Testing your setup...")
print("=" * 40)

# Test 1: Check packages
print("1. Checking installed packages:")
try:
    print(f"   OpenCV version: {cv2.__version__}")
except:
    print("   ✗ OpenCV not working")

try:
    print(f"   NumPy version: {np.__version__}")
except:
    print("   ✗ NumPy not working")

# Test 2: Check model files
print("\n2. Checking model files:")
model_files = ["yolov4-tiny.weights", "yolov4-tiny.cfg", "coco.names"]
all_ok = True

for file in model_files:
    if os.path.exists(f"models/{file}"):
        print(f"   ✓ {file} found")
    else:
        print(f"   ✗ {file} missing")
        all_ok = False

# Test 3: Check webcam
print("\n3. Testing webcam:")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("   ✓ Webcam is accessible")
    ret, frame = cap.read()
    if ret:
        print(f"   ✓ Can read frames (size: {frame.shape})")
    else:
        print("   ✗ Cannot read frames from webcam")
    cap.release()
else:
    print("   ✗ Cannot access webcam")

print("=" * 40)
if all_ok:
    print("✓ All tests passed! You're ready for object detection!")
else:
    print("✗ Some tests failed. Please check the errors above.")