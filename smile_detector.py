#!/usr/bin/python3

import cv2
from picamera2 import Picamera2

# Load the pre-trained Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")

# Load the pre-trained Haar cascade classifier for smile detection
smile_cascade = cv2.CascadeClassifier("/usr/share/opencv4/haarcascades/haarcascade_smile.xml")

# Initialize the Picamera2 object
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Initialize variables for smile count and log file
smile_count = 0
log_file = open("smile_log.txt", "a")

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # For each face detected, check for smiles
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the region of interest (ROI) in the grayscale frame
        roi_gray = gray[y:y+h, x:x+w]
        # Get the region of interest (ROI) in the color frame
        roi_color = frame[y:y+h, x:x+w]

        # Detect smiles in the ROI of the face
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22)

        # If a smile is detected, draw a rectangle around it and update smile count
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
            # Increment smile count
            smile_count += 1
            # Write smile detection to log file
            log_file.write("Smile detected!\n")

    # Display the frame
    cv2.imshow('Smile Detector', frame)

    # Check for the 'q' key to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera object and close all OpenCV windows
picam2.stop()
cv2.destroyAllWindows()

# Write smile count to log file
log_file.write(f"Total number of smiles detected: {smile_count}\n")
log_file.close()
