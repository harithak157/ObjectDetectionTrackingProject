import cv2
import urllib.request
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Files we need to download
files = {
    "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
    "yolov4-tiny.cfg": "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg", 
    "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

print("Starting download of model files...")

# Download each file
for filename, url in files.items():
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, f"models/{filename}")
        print(f"✓ Successfully downloaded {filename}")
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")

print("Download process completed!")