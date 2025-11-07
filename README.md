# Driver Drowsiness Detection System (CNN)

This project is a real-time driver drowsiness detection system built with Python, OpenCV, and a custom-trained TensorFlow/Keras deep learning model. It uses a webcam to monitor a driver's eyes and sounds an audible alarm if it detects signs of fatigue.

## Features
* **Real-Time Detection:** Uses OpenCV to process the webcam feed at a high frame rate.
* **AI-Powered:** A custom-trained Convolutional Neural Network (CNN) classifies eye states as "Open" or "Closed" with **~99% accuracy**.
* **Smart Alarm Logic:** Implements a robust "score" system to differentiate between a normal blink and a dangerous, prolonged drowsy closure.
* **Audible Alert:** Plays an alarm sound to alert the driver when the drowsiness score exceeds a safe threshold.

---
## How It Works

This project is built in two distinct phases:
1.  **Phase 1: Model Training (The "Brain")** - We build and train an AI model to be an expert on eyes.
2.  **Phase 2: Real-Time Detection (The "App")** - We build an application that uses the "brain" on a live video feed.

### Phase 1: Model Training (The "Brain")

The "brain" of this project is a **Convolutional Neural Network (CNN)**. A CNN is a type of AI model that is specifically designed to "see" and find patterns in images.

#### 1. Data Preparation (`prepare_data.py`)
To train our "brain," we first need a lot of examples. We used the **MRL Eye Dataset**, which contains over 80,000 images of open and closed eyes. Before we can use them, we must prepare them:
* **Grayscale:** We convert images to black and white. Color isn't needed to see if an eye is open or closed, and this makes training 3x faster.
* **Resizing:** Our model needs all images to be the *exact same size*. We resized all images to `80x80` pixels.
* **Normalization:** This is a key technique. A computer sees a pixel as a number from 0 (black) to 255 (white). We scale these numbers to be between `0.0` and `1.0` by dividing by 255. This makes the model train much faster and more reliably.

#### 2. Model Training (`train_model.py`)
This script defines the CNN's architecture (its layers) and trains it on the prepared data.
* The model "looks" at thousands of images and "guesses" if the eye is open or closed.
* After each guess, it checks the right answer and slightly adjusts its internal logic (its "filters") to be more accurate.
* We run this process for 10 "epochs" (10 full passes over the entire dataset).
* The final, trained "brain" is then saved to the `saved_model/drowsiness_model.h5` file.

#### Model Training Results
Our model was trained for 10 epochs and achieved **~99% validation accuracy**. The plots below show that the model learned effectively without overfitting (where the training and validation lines diverge).
<img width="1919" height="1079" alt="Screenshot 2025-11-06 035328" src="https://github.com/user-attachments/assets/179b0c2a-ad4e-4667-be21-060403492d43" />

<img width="1919" height="1079" alt="Screenshot 2025-11-06 035228" src="https://github.com/user-attachments/assets/24da238d-a182-4c9d-b5c1-d779c20fc5e5" />

---
### Phase 2: Real-Time Detection (The "App")

This is the `run_detection.py` script, which ties everything together. It runs in a continuous `while True:` loop, performing these steps on every single frame from your webcam:

#### 1. Find the Face (Haar Cascades)
It's slow to search the entire video frame for tiny eyes. Instead, we first find the face using a **Haar Cascade** (`assets/haar_cascade_frontalface_alt.xml`). This is a very fast, pre-trained detector that finds the general pattern of a human face.

#### 2. Find the Eyes (Haar Cascades)
Once we have the "box" for the face, we search *only inside that box* for the eyes, using two more Haar Cascades (`assets/haarcascade_lefteye_2splits.xml` and `assets/haarcascade_righteye_2splits.xml`).

This two-step "cascade" (find face first, *then* find eyes) is a key technique that makes the program fast and efficient.

#### 3. Predict with our "Brain"
* We cut out the small eye image, resize it to `80x80`, and normalize it.
* We feed this image to our loaded `drowsiness_model.h5`.
* The model gives us a prediction: a number close to `0` (Closed) or `1` (Open).

#### 4. The "Score" Logic (The Most Important Part)
We can't sound an alarm every time the model sees a "Closed" eye, because the driver needs to blink! We need to tell the difference between a **fast blink** and a **slow, drowsy closure**.

We do this with a **score** system:
* If the model predicts **"Closed"**: We add 1 to the score (`score += 1`).
* If the model predicts **"Open"**: We subtract 1 from the score (`score -= 1`).

This is brilliant because:
* **A fast blink:** The score might go `+1, +1` and then immediately ` -1, -1`. It stays low and never reaches the alarm.
* **A drowsy driver:** The eyes close and stay closed. The score climbs: `+1, +2, +3... +10... +15...`.
* When the score passes our `ALARM_THRESHOLD` (set to `5` for easy demonstration), we sound the alarm!

---
## Project Structure
```
Driver-Drowsiness-Detection-CNN/
├── assets/
│   ├── haarcascade_frontalface_alt.xml      # Finds faces
│   ├── haarcascade_lefteye_2splits.xml      # Finds left eyes
│   ├── haarcascade_righteye_2splits.xml     # Finds right eyes
│   └── alarm.wav                            # The alarm sound
│
├── saved_model/
│   └── drowsiness_model.h5                  # The trained "brain"
│
├── .gitignore                               # Tells Git what to ignore
├── prepare_data.py                          # Script to process the dataset
├── train_model.py                           # Script to train the model
├── run_detection.py                         # The main application to run
├── requirements.txt                         # List of Python libraries
└── README.md                                # This file!
```
---

## Installation & Usage 
This guide is for anyone who wants to run the detection app. 
### 1. Get the Code 
Clone this repository to your local machine:
```bash 
git clone [https://github.com/verciel/Driver-Drowsiness-Detection-CNN.git](https://github.com/Your-Username/Driver-Drowsiness-Detection-CNN.git) cd Driver-Drowsiness-Detection-CNN
```

### 2. Set Up The Environment (For Users)
1. Create a virtual environment:
`python -m venv myenv`

2. Activate it:
- On Windows: `myenv\Scripts\activate`
- On macOS/Linux: `source myenv/bin/activate`

3. Install all required libraries:
`pip install -r requirements.txt`

4. Run the Application!
With your webcam on and your environment active, simply run:
`python run_detection.py`
 Press **'q'** to quit the application.
 
---
## Limitations

- **Glasses:** The system performs poorly with glasses. Glare on the lenses confuses the Haar Cascades and the CNN, preventing them from finding the eye.

- **Lighting:** The model requires good, even lighting. In very dark or very bright conditions, the webcam image is not clear enough for the detectors to work. 

Real-world systems in cars solve this by using **Infrared (IR) cameras**, which can see your eyes perfectly even in the dark or with glasses on.

---
