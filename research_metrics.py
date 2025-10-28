import cv2
import numpy as np
import time
import csv
from datetime import datetime

# Initialize detection system
net = cv2.dnn.readNetFromDarknet("models/yolov4-tiny.cfg", "models/yolov4-tiny.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open("models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create CSV file for storing results
results_file = f"research_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(results_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Frame", "Processing_Time", "FPS", "Objects_Detected"])

frame_count = 0
print("Collecting data for research paper. Press 'q' to stop...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    height, width, channels = frame.shape

    # Prepare and process image
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    start_time = time.time()
    outs = net.forward(output_layers)
    end_time = time.time()
    
    processing_time = end_time - start_time
    fps = 1.0 / processing_time
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Count unique objects detected
    objects_detected = set()
    if len(indices) > 0:
        for i in indices.flatten():
            objects_detected.add(classes[class_ids[i]])
    
    # Save results to CSV
    with open(results_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            frame_count,
            f"{processing_time:.4f}",
            f"{fps:.2f}",
            len(objects_detected)
        ])
    
    # Display frame with info
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Objects: {len(objects_detected)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Research Data Collection", frame)
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"Data collection complete. Results saved to {results_file}")
print("You can use this data for your research paper analysis.")