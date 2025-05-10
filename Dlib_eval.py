import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import os
from sklearn.metrics import confusion_matrix, f1_score

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

# Path to your test data directory
test_data_dir = r"F:\machine_learning_studying\jupyter\graduation_project\Drowsiness\drowsiness-dlib\tests\datatest\test"  # Update with your path

# Define thresholds
EAR_THRESHOLD = 0.17  
YAWN_THRESHOLD = 0.75  

# Initialize counters and lists to store results
drowsy_count = 0
yawning_count = 0
drowsy_images = []
yawning_images = []

# Initialize lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Loop through test images
for class_name in os.listdir(test_data_dir):  # 'alert' and 'drowsy' folders
    class_dir = os.path.join(test_data_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)

        # Load image
        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
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

            # Check for drowsiness
            if ear < EAR_THRESHOLD:
                drowsy_count += 1
                drowsy_images.append(image_path)
                predicted_labels.append('drowsy')
            else:
                predicted_labels.append('alert')

            # Check for yawning
            if mor > YAWN_THRESHOLD:
                yawning_count += 1
                yawning_images.append(image_path)

            # Append true label
            true_labels.append(class_name)

# Print results
print("Total drowsy images:", drowsy_count)
print("Total yawning images:", yawning_count)
print("Drowsy images:", drowsy_images)
print("Yawning images:", yawning_images)

# Calculate confusion matrix and F1 score
conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=['alert', 'drowsy'])
f1 = f1_score(true_labels, predicted_labels, pos_label='drowsy')

# Print confusion matrix and F1 score
print("Confusion Matrix:")
print(conf_matrix)
print("F1 Score:", f1)