import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random

# --- 1. Define Constants ---
DATA_DIR = 'mrlEyes_2018_01'  # Folder containing the category folders
IMG_SIZE = 80                 # We will resize all images to 80x80
CATEGORIES = ["Close-Eyes", "Open-Eyes"] # 0 = closed, 1 = open

# --- 2. Create Training Data ---
print("Starting data preparation...")
training_data = []

def create_training_data():
    for category in CATEGORIES:  # This will be "closed eyes", then "open eyes"
        path = os.path.join(DATA_DIR, category)
        label = CATEGORIES.index(category)  # 0 for "closed eyes", 1 for "open eyes"

        print(f"Loading category: {category} (Label: {label})")

        for img in os.listdir(path):
            try:
                # Read the image
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Read as grayscale

                # Resize the image
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                # Add to our data
                training_data.append([new_array, label])

            except Exception as e:
                # Some files might be broken, just skip them
                print(f"Error reading {img_path}: {e}")
                pass

create_training_data()
print(f"Data creation complete. Total images: {len(training_data)}")

# --- 3. Shuffle and Separate ---
random.shuffle(training_data)

X = []  # This will hold the image data
y = []  # This will hold the labels

for features, label in training_data:
    X.append(features)
    y.append(label)

# --- 4. Reshape and Normalize ---
# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Reshape X to fit the model (Keras needs one more dimension for "channels")
# (Num_images, IMG_SIZE, IMG_SIZE, 1) -> 1 is for grayscale
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Normalize the image data (pixels from 0-255 to 0-1)
X = X / 255.0
X = X.astype(np.float32)
# --- 5. Split into Training and Testing sets ---
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# --- 6. Save the Processed Data ---
print("Saving processed data to files...")
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("All done!")