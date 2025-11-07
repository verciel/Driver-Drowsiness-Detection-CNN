import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# --- 1. Load the Prepared Data ---
print("Loading prepared data...")
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print("Data loaded successfully.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Define image size (must match prepare_data.py)
IMG_SIZE = 80

# --- 2. Define the CNN Model ---
print("Building the model...")
model = Sequential()

# Layer 1: Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D((2, 2)))

# Layer 2: Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Layer 3: Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Layer 4: Flatten Layer
model.add(Flatten())

# Layer 5: Dense Layer (Full connection)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5)) # Dropout to prevent overfitting

# Layer 6: Output Layer
# Use 'sigmoid' for binary (2-class) classification
model.add(Dense(1, activation='sigmoid'))

# --- 3. Compile the Model ---
print("Compiling the model...")
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 4. Train the Model ---
print("Training the model...")
# An 'epoch' is one full pass through the entire training dataset
# 'validation_data' checks the model's performance on the unseen test data after each epoch
history = model.fit(X_train, y_train, 
                    epochs=10, 
                    batch_size=32, 
                    validation_data=(X_test, y_test))

# --- 5. Save the Trained Model ---
print("Saving the trained model...")
model.save('drowsiness_model.h5')
print("Model saved as drowsiness_model.h5")

# --- 6. (Optional) Plot Training History ---
print("Plotting training history...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print("All done!")