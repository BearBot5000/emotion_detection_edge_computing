#!/home/mizebrent/helpme/bin/python

import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import os
import time

# Update the model path if necessary
model_path = '/home/mizebrent/emotion/emotion_detection_model_tuned.tflite'

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Load face cascade classifier
face_cascade_path = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'  # Update this path if necessary
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Initialize Picamera2 and configure it
picam2 = Picamera2()
picam2.start_preview()
config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
picam2.configure(config)

# Create a directory to save preprocessed face images
os.makedirs("inference_faces", exist_ok=True)
face_count = 0

# Video settings
video_duration = 30  # seconds
frame_width = 640
frame_height = 480
fps = 20

# Initialize video writer
video_writer = cv2.VideoWriter('emotion_detection_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

try:
    picam2.start()

    def picamera2_to_cv2(picam2_image):
        return cv2.cvtColor(picam2_image, cv2.COLOR_BGRA2BGR)

    start_time = time.time()
    frame_count = 0

    while True:
        # Capture frame from Picamera2
        frame = picamera2_to_cv2(picam2.capture_array())

        # Flip the frame horizontally
        frame = cv2.flip(frame, -1)

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = gray_frame[y:y+h, x:x+w]

            # Resize the face ROI to match the input shape of the model
            resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

            # Normalize the resized face image
            normalized_face = resized_face / 255.0

            # Reshape the image to match the input shape of the model
            reshaped_face = np.expand_dims(normalized_face, axis=0)
            reshaped_face = np.expand_dims(reshaped_face, axis=-1).astype(np.float32)

            # Save the preprocessed face image for verification
            cv2.imwrite(f"inference_faces/face_{face_count}.png", resized_face * 255)
            face_count += 1

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], reshaped_face)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            print("Predictions:", preds)  # Debugging line to check predictions

            emotion_idx = np.argmax(preds)
            emotion = emotion_labels[emotion_idx]

            print("Detected emotion:", emotion)  # Debugging line to check detected emotion

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Write the frame to the video file
        video_writer.write(frame)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)
        frame_count += 1

        # Calculate and print FPS every 30 frames
        if frame_count == 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"FPS: {fps}")
            frame_count = 0
            start_time = time.time()

        # Check if video duration is reached
        if time.time() - start_time > video_duration:
            break

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the Picamera2 object and close all windows
    picam2.stop()
    cv2.destroyAllWindows()
    video_writer.release()