import cv2
import os
import numpy as np

# Paths
haar_cascade_path = 'src/haarcascade_frontalface_alt.xml'
model_path = 'trained_model.yml'
faces_dir = 'faces'

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(haar_cascade_path)
if face_cascade.empty():
    print("❌ Haar cascade not loaded properly!")
    exit()

# Load trained recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_path)

# Get names from faces folder
names = sorted(os.listdir(faces_dir))

# Start webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Cannot open camera!")
    exit()

print("✅ Face recognition started! Press 'q' to quit.")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (200, 200))

        label, confidence = recognizer.predict(face_roi)

        if confidence < 70:  # lower = more confident
            name = names[label - 1]  # our labels start from 1
            text = f"{name} ({confidence:.0f})"
            color = (0, 255, 0)
        else:
            text = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
print("👋 Exiting face recognition.")
