import cv2
import sys

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)  # 0 is the default camera on most systems, you can change it if needed

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error capturing frame")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a green rectangle around detected faces with confidence scores
    for (x, y, w, h) in faces:
        # Calculate confidence score
        confidence = 100 - ((w * h) / (frame.shape[0] * frame.shape[1])) * 100
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Face: {confidence:.2f}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with the green rectangles
    cv2.imshow('Face Detection', frame)
        # Check for the "stop" command in the terminal
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break
# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


