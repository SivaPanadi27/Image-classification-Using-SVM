import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the preprocessed data from the pickle file
pickle_file = "data1.pickle"
with open(pickle_file, 'rb') as pick_in:
    data = pickle.load(pick_in)

print(f"Data loaded successfully from {pickle_file}")

# Step 2: Separate features (X) and labels (y)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

# Convert lists to NumPy arrays
X = np.array(features)
y = np.array(labels)

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Step 4: Train the SVM model
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
categories = ['Animals', 'Objects', 'Landscapes']  # Update categories to match your project
report = classification_report(y_test, y_pred, target_names=categories)
print("Classification Report:")
print(report)

# Step 6: Visualize a sample prediction
index = 0  # Index of the first test sample
prediction = model.predict([X_test[index]])[0]
true_label = y_test[index]

print(f"True Label: {categories[true_label]}")
print(f"Predicted Label: {categories[prediction]}")

# Reshape the flattened image back to 2D
image = X_test[index].reshape(50, 50)  # Use the same img_size used during preprocessing
plt.imshow(image, cmap='gray')
plt.title(f"True: {categories[true_label]}, Predicted: {categories[prediction]}")
plt.axis('off')
plt.show()
