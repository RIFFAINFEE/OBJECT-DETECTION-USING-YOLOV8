Source Code

import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # YOLOv8 Nano model


cap = cv2.VideoCapture(0)

while cap.isOpened():
ret, frame = cap.read()
if not ret:
break    
results = model(frame)  # Run YOLO
annotated_frame = results[0].plot()  # Draw detections

cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
break

cap.release()
cv2.destroyAllWindows()