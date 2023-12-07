import cv2
import numpy as np
from openpose.examples.tutorial_api_python import openpose_python as op

# Custom patshs for OpenPose
params = dict()
params["model_folder"] = "/Users/ahilankaruppusami/Coding Projects/AI Image Detection/openpose/models"

# Initialize OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Open the camera
cap = cv2.VideoCapture(0)

# Initialize count and head_detected variables
count = 0
head_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame with OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # Get the keypoints for the person in the frame
    if datum.poseKeypoints.shape:
        keypoints = datum.poseKeypoints[0]

        # Check if the head is detected in the top view
        if keypoints[0][1] > 0 and keypoints[0][0] > 0:
            head_detected = True
        else:
            head_detected = False

        # Update the count based on the head's movement
        if head_detected:
            if keypoints[0][0] > frame.shape[1] / 2:
                count += 1
            elif keypoints[0][0] < frame.shape[1] / 2:
                count -= 1

    # Overlay the count on the video feed
    cv2.putText(frame, f"Count: {count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the live video feed
    cv2.imshow("Top-View Head Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the camera, OpenPose, and close the OpenCV window
cap.release()
opWrapper.stop()
cv2.destroyAllWindows()
