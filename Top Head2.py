import cv2
import numpy as np

# Load the SSD model with your file paths
net = cv2.dnn.readNetFromCaffe('/Users/ahilankaruppusami/Coding_Projects/AI_Image_Detection/opencv/samples/dnn/face_detector/deploy.prototxt',
                               '/Users/ahilankaruppusami/Coding_Projects/AI_Image_Detection/MobileNet-SSD-master/mobilenet_iter_73000.caffemodel')

# Initialize the video stream
cap = cv2.VideoCapture(0)  # Use '0' to access the default camera

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize the frame for faster processing
    frame = cv2.resize(frame, (300, 300))

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the input for the neural network
    net.setInput(blob)
    
    # Forward pass through the layers
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            # Get the coordinates of the top left and bottom right corners of the bounding box
            startX = int(detections[0, 0, i, 3] * frame.shape[1])
            startY = int(detections[0, 0, i, 4] * frame.shape[0])
            endX = int(detections[0, 0, i, 5] * frame.shape[1])
            endY = int(detections[0, 0, i, 6] * frame.shape[0])

            # Draw a green box around the detected object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
