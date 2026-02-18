import cv2
import os
import numpy as np

# Path to the faces dataset
dataset_path = "faces"

# Create lists for images and labels
faces = []
labels = []
names = {}
current_id = 0

# Loop through all subfolders (each person)
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    names[current_id] = person_name

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (200, 200))
        faces.append(img)
        labels.append(current_id)

    current_id += 1

# Convert to numpy arrays
faces = np.array(faces)
labels = np.array(labels)

# Create and train LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

# Save model and label map
recognizer.save("trained_model.yml")

# Save label names
with open("labels.txt", "w") as f:
    for label_id, name in names.items():
        f.write(f"{label_id},{name}\n")

print("✅ Training complete! Model saved as 'trained_model.yml'")
print("✅ Labels saved as 'labels.txt'")
