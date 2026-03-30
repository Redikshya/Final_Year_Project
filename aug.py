import os
import random
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

# Path to the main faces folder
BASE_DIR = "static/faces"

def augment_image(img):
    """Apply random augmentations: brightness, rotation, flip, noise"""
    img = img.convert("RGB")

    # Random rotation between -20° and +20°
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, expand=True)

    # Random horizontal flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    
    return img

def augment_folder(folder_path):
    """Randomly select 20 images from folder and overwrite them with augmented versions"""
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(images) < 20:
        print(f"Skipping {folder_path}: not enough images (<30)")
        return

    selected = random.sample(images, 20)

    for img_name in selected:
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path)
            aug_img = augment_image(img)
            aug_img.save(img_path)  # overwrite the original file
        except Exception as e:
            print(f"Error processing {img_path}: {e}")


def main():
    """Iterate through all username$id$ folders and apply augmentation"""
    for folder in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder)
        if os.path.isdir(folder_path):
            print(f"Augmenting images in: {folder_path}")
            augment_folder(folder_path)
    print("Augmentation completed successfully.")


if __name__ == "__main__":
    main()
