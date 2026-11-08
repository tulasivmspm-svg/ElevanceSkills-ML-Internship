
import cv2
import os
import json
import numpy as np

# Folder containing registered student images
DATASET_PATH = "student_data"

# Output model location
MODEL_PATH = "models/student_face_model.yml"
LABELS_PATH = "models/student_labels.json"

faces = []
labels = []
student_labels = {}

current_label = 0

# Check student data folder
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("student_data folder not found!")

# Read each student's folder
for student_id in sorted(os.listdir(DATASET_PATH)):

    student_folder = os.path.join(DATASET_PATH, student_id)

    if not os.path.isdir(student_folder):
        continue

    student_labels[current_label] = student_id
    image_count = 0

    for filename in os.listdir(student_folder):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(student_folder, filename)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print("Could not read:", image_path)
            continue

        # Ensure all images have the same dimensions
        image = cv2.resize(image, (200, 200))

        faces.append(image)
        labels.append(current_label)

        image_count += 1

    print(f"{student_id}: {image_count} images loaded")

    current_label += 1

# Ensure training data exists
if len(faces) < 2:
    raise ValueError("At least two face images are required!")

# Convert labels into NumPy array
labels = np.array(labels, dtype=np.int32)

# Create LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train recognizer
recognizer.train(faces, labels)

# Create models folder
os.makedirs("models", exist_ok=True)

# Save trained model
recognizer.save(MODEL_PATH)

# Save student ID mapping
with open(LABELS_PATH, "w") as file:
    json.dump(student_labels, file, indent=4)

print("\n================================")
print("FACE MODEL TRAINING COMPLETED!")
print("================================")
print("Total training images:", len(faces))
print("Registered students:", len(student_labels))
print("Model saved:", MODEL_PATH)
print("Student labels saved:", LABELS_PATH)
