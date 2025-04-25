import os
import numpy as np
import cv2

# Step 1: Define the directory where the images are saved
dir = r"C:\Users\HP\Desktop\DATASET"  # Replace with your dataset directory

# Step 2: Define the categories (folders)
categories = ['Animals', 'Objects', 'Landscapes']  # Folder names

# Step 3: Initialize lists for features and labels
X = []  # Feature vectors (flattened images)
y = []  # Labels

# Step 4: Define image size for resizing
img_size = (50, 50)  # You can change this size as needed

# Step 5: Loop through each category and process images
for category in categories:
    path = os.path.join(dir, category)  # Path to the category folder
    label = categories.index(category)  # Numeric label for the category

    print(f"Processing images in folder: {category}")
    for img_name in os.listdir(path):  # List all images in the folder
        img_path = os.path.join(path, img_name)  # Full path to the image

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
            X.append(img_flat)
            y.append(label)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Step 6: Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)

# Step 7: Print summary
print(f"Total images processed: {len(X)}")
