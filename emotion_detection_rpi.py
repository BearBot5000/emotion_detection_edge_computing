#!/home/mizebrent/helpme/bin/python

import cv2
import numpy as np
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite

# Update the model path if necessary
model_path = '/home/mizebrent/emotion/emotion_detection_model.tflite'

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

try:
    picam2.start()

    def picamera2_to_cv2(picam2_image):
        return cv2.cvtColor(picam2_image, cv2.COLOR_RGB2BGR)

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

            # Perform inference
            interpreter.set_tensor(input_details[0]['index'], reshaped_face)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            emotion_idx = np.argmax(preds)
            emotion = emotion_labels[emotion_idx]

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Real-time Emotion Detection', frame)
        cv2.waitKey(1)
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release the Picamera2 object and close all windows
    picam2.stop()
    cv2.destroyAllWindows()
e
