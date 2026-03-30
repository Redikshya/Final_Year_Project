import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import pickle

# Load your trained model
model = load_model('face_recognition_model.h5')  # Replace with your model path

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Get the list of class names from the label encoder
class_names = label_encoder.classes_

# Print the class names to verify
print("Class names:", class_names)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a confidence threshold
confidence_threshold = 0.90 # Adjust this threshold as needed

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Preprocess the face for prediction
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  # Convert detected face to RGB
        face_roi = cv2.resize(face_roi, (128, 128))  # Replace with your input dimensions
        face_roi = face_roi.astype('float32') / 255.0  # Normalize the pixel values
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(face_roi)
        max_confidence = np.max(predictions)  # Get the highest confidence score
        label_index = np.argmax(predictions)  # Get the index of the highest score

        # Determine if the detected face is recognized or unknown
        if max_confidence >= confidence_threshold:
            label = label_encoder.inverse_transform([label_index])[0]  # Decode label
        else:
            label = "Unknown"  # Set label as "Unknown" for low confidence predictions

        # Put the label text on the frame
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()



