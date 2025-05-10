import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import time

# Load pre-trained dlib models for face detection and landmark prediction
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("F:\machine_learning_studying\jupyter\graduation_project\Drowsiness\drowsiness-dlib\src\models\shape_predictor_68_face_landmarks.dat")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Opening Ratio (MOR)
def yawn_detection(mouth):
    A = dist.euclidean(mouth[3], mouth[9])  # Vertical distance
    B = dist.euclidean(mouth[0], mouth[6])  # Horizontal distance
    yawn = A / B
    return yawn

# Initialize camera
cap = cv2.VideoCapture(0)

# Define thresholds for EAR and MOR
EAR_THRESHOLD = 0.17  # Threshold for eye aspect ratio to detect drowsiness
YAWN_THRESHOLD = 0.75  # Threshold for mouth opening ratio to detect yawning

# Initialize counters and flags
EAR_COUNTER = 0  # Counter for consecutive frames with EAR below threshold
YAWN_COUNTER = 0  # Counter for consecutive yawns
ALARM_ON = False  # Flag to indicate if the alarm is on
ear_start_time = time.time()  # Start time for EAR check
mor_start_time = time.time()  # Start time for MOR check

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(gray, face)
        landmarks = [(p.x, p.y) for p in landmarks.parts()]

        # Get the coordinates for the eyes and mouth
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]

        # Calculate EAR and MOR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mor = yawn_detection(mouth)

        # Check EAR every 5 seconds
        if time.time() - ear_start_time >= 3:
            if ear < EAR_THRESHOLD:
                print("Drowsiness detected!")
                cv2.putText(frame, "Drowsiness detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ear_start_time = time.time()  # Reset the timer

        # Check MOR every 3 seconds
        if time.time() - mor_start_time >= 3:
            if mor > YAWN_THRESHOLD:
                YAWN_COUNTER += 1
                print("Yawn detected number: ", YAWN_COUNTER)
                if YAWN_COUNTER >= 3:
                    print("Warning: Frequent yawning detected!")
                    cv2.putText(frame, "Warning: Frequent yawning detected!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    YAWN_COUNTER = 0  # Reset the counter
            mor_start_time = time.time()  # Reset the timer

        # Draw landmarks on the eyes and mouth
        for (x, y) in left_eye + right_eye + mouth:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Display the video
    cv2.imshow("Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()