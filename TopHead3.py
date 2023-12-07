import cv2
import numpy as np

# Load the MobileNet-SSD model
net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v1_coco.pbtxt')

# Initialize the camera or video stream
cap = cv2.VideoCapture(0)  # Use '0' for the built-in Mac camera.

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for detection in detections[0, 0, :, :]:
        confidence = detection[2]
        class_id = int(detection[1])

        if confidence > 0.5 and class_id == 15:  # Class ID 15 corresponds to 'person'
            # Do something when a person is detected (e.g., draw a bounding box or trigger an alert).
            # You can also track the person's movement over multiple frames.
            # Example: Draw a bounding box around the detected person
            x = int(detection[3] * frame.shape[1])
            y = int(detection[4] * frame.shape[0])
            w = int((detection[5] - detection[3]) * frame.shape[1])
            h = int((detection[6] - detection[4]) * frame.shape[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
