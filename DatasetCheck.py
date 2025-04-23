import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

dir = r"C:\Users\HP\Desktop\DATASET"
categories = ['Animals', 'Objects', 'Landscapes']

for category in categories:
    path = os.path.join(dir, category)

    print(f"Processing images in folder: {category}")
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)

        try:
            img = cv2.imread(img_path, 1)
            if img is None:
                print(f"Error loading image: {img_path}")
                continue

            cv2.imshow(f"Image from {category}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    break
