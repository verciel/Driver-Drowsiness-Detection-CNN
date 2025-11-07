import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pygame

# --- 1. Load All Assets ---
print("Loading assets...")
# Load the pre-trained Keras model
model = load_model('saved_model/drowsiness_model.h5')

# Load the Haar cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_alt.xml')
left_eye_cascade = cv2.CascadeClassifier('assets/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier('assets/haarcascade_righteye_2splits.xml')

# Initialize Pygame for playing the alarm sound
pygame.mixer.init()
pygame.mixer.music.load('assets/alarm.wav') # Make sure you have an 'alarm.wav' file
print("Assets loaded.")

# --- 2. Constants and Variables ---
IMG_SIZE = 80  # Must match the size used during training
ALARM_THRESHOLD = 5  # Score threshold to trigger alarm
score = 0             # The drowsiness score
alarm_playing = False

# Labels for prediction
labels = ['Closed', 'Open']

# --- 3. Start Video Capture ---
cap = cv2.VideoCapture(0) # 0 is for the default webcam
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting webcam. Press 'q' to quit.")

# --- 4. Main Detection Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- ALL THE CODE BELOW THIS LINE IS NOW CORRECTLY INDENTED ---

    # Convert the frame to grayscale for cascade detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        # No face detected, reset score and stop alarm
        if alarm_playing:
            pygame.mixer.music.stop()
            alarm_playing = False
        score = 0
    
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Get the region of interest (ROI) for the face
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect left eye
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
        # Detect right eye
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray)

        eye_detected = False
        
        # Combine both eye detections
        for (ex, ey, ew, eh) in list(left_eyes) + list(right_eyes):
            if eye_detected: # Only process one eye per frame
                break

            # Get the eye ROI
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            
            # --- Pre-process the eye image ---
            eye_roi = cv2.resize(eye_roi, (IMG_SIZE, IMG_SIZE))
            eye_roi = eye_roi / 255.0  # Normalize
            eye_roi = eye_roi.astype(np.float32)
            eye_roi = eye_roi.reshape(-1, IMG_SIZE, IMG_SIZE, 1) # Reshape for model

            # --- Predict with the model ---
            prediction = model.predict(eye_roi)
            
            # prediction[0][0] will be a value: ~0 for 'Closed', ~1 for 'Open'
            if prediction[0][0] > 0.5:
                status = 'Open'
                score -= 1  # Subtract from score
                if score < 0:
                    score = 0 # Don't go below 0
            else:
                status = 'Closed'
                score += 1 # Add to score

            # Display the status
            cv2.putText(frame, f"Eye: {status}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_detected = True

        # --- Handle Drowsiness Logic ---
        if score > ALARM_THRESHOLD:
            if not alarm_playing:
                pygame.mixer.music.play(-1) # Play on loop
                alarm_playing = True
            
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False
        
        if not eye_detected:
            # If a face is found but no eye, reset score
            score = 0
            if alarm_playing:
                pygame.mixer.music.stop()
                alarm_playing = False
    
    # Display the score on the frame
    cv2.putText(frame, f"Score: {score}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Drowsiness Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Cleanup ---
print("Cleaning up...")
cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()