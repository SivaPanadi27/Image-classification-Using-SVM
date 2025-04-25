import os
import numpy as np
import cv2
import pickle

# Step 1: Define the directory where the images are saved
dir = r"C:\Users\HP\Desktop\DATASET"  # Replace with your dataset directory

# Step 2: Define the categories (folders)
categories = ['Animals', 'Objects', 'Landscapes']  # Folder names

# Step 3: Initialize a list to store the data
data = []

# Step 4: Define image size for resizing
img_size = (50, 50)  # You can change this size as needed

# Step 5: Loop through each category and process images
for category in categories:
    path = os.path.join(dir, category)  # Path to the category folder
    label = categories.index(category)  # Numeric label for the category

    print(f"Processing images in folder: {category}")
    for img_name in os.listdir(path):  # List all files in the folder
        img_path = os.path.join(path, img_name)  # Full path to the file

        # Skip non-image files (e.g., _DS_Store)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            print(f"Skipping non-image file: {img_name}")
            continue

        try:
            # Load the image in grayscale
            img = cv2.imread(img_path, 0)
            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            # Resize the image
            img_resized = cv2.resize(img, img_size)

            # Flatten the image into a 1D array
            img_flat = img_resized.flatten()

            # Normalize pixel values to [0, 1]
            img_flat = img_flat / 255.0

            # Append the flattened image and its label
            data.append([img_flat, label])

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Step 6: Save the processed data to a pickle file
pickle_file = "data1.pickle"
with open(pickle_file, 'wb') as pick_in:
    pickle.dump(data, pick_in)

print(f"Data saved to {pickle_file}")
