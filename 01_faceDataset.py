import cv2
import os
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Define the folder path as a variable
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
dataset_folder_path = os.path.join(BASE_FOLDER, "dataset")

def load_images_from_folder(folder_path, img_size=(128, 128)):
    images = []       # To store the processed images
    labels = []       # To store corresponding labels (names)

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path)

                if img is not None:
                    # --- Resize with padding (keep aspect ratio) ---
                    h, w = img.shape[:2]
                    desired_w, desired_h = img_size
                    scale = min(desired_w / w, desired_h / h)
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized = cv2.resize(img, (new_w, new_h))

                    # Create a black background and paste centered image
                    padded = np.zeros((desired_h, desired_w, 3), dtype=np.uint8)
                    x_offset = (desired_w - new_w) // 2
                    y_offset = (desired_h - new_h) // 2
                    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

                    img = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
                    # ----------------------------------------------

                    images.append(img)
                    labels.append(label)
    
    return np.array(images), np.array(labels)

def preprocess_and_save(folder_path, output_path='data.pkl', img_size=(128, 128)):
    images, labels = load_images_from_folder(folder_path, img_size)
    images = images / 255.0

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    with open(output_path, 'wb') as f:
        pickle.dump((x_train, x_test, y_train, y_test, le), f)
    
    print("Data preprocessed and saved to", output_path)

def check_labels_and_counts(folder_path):
    label_counts = {}
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            num_images = len([f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))])
            label_counts[label] = num_images
    return label_counts

preprocess_and_save(dataset_folder_path)

label_counts = check_labels_and_counts(dataset_folder_path)
print("Labels and their image counts:")
for label, count in label_counts.items():
    print(f"{label}: {count} images")
