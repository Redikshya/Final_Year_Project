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

# Define the path to the new image
image_path = 'C:/Users/nphuy/OneDrive/Desktop/predict/pp_face0.jpg'  # Replace with the path to the new image
image = cv2.imread(image_path)

# Check if the image is loaded
if image is None:
    print("Error: Image not found.")
else:
    # Preprocess the image
    face_roi = cv2.resize(image, (128, 128))  # Replace with your input dimensions
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)  # Convert to RGB
    face_roi = face_roi.astype('float32') / 255.0  # Normalize the pixel values
    face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(face_roi)
    label_index = np.argmax(predictions)  # Get the index of the highest score
    label = label_encoder.inverse_transform([label_index])[0]  # Decode label

    # Print the predicted name
    print("Predicted Name:", label)

    # Optionally display the image with the prediction
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Predicted Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
