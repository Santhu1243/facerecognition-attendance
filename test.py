import os
import cv2
import face_recognition
import numpy as np
from datetime import datetime

# Set OpenCV backend to avoid GUI errors
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Define paths for known faces and captured images
KNOWN_FACES_FOLDER = "/home/chezzuser/Desktop/face_auth/facerecognition-attendance/known_faces"  # Change if needed
SAVE_FOLDER = "/home/chezzuser/Desktop/face_auth/facerecognition-attendance/captured_images"

# Ensure save folder exists
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Load known faces
known_face_encodings = []
known_face_names = []

for filename in os.listdir(KNOWN_FACES_FOLDER):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(KNOWN_FACES_FOLDER, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:  # Ensure at least one encoding exists
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Extract name from filename

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Use V4L2 for better performance

# Set lower resolution for speed optimization
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Failed to access the camera. Try changing the camera index.")
    exit()

print("Face recognition started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame. Exiting...")
        break

    # Convert frame to RGB (face_recognition requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)  # Lower tolerance for better accuracy
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_face_names[matched_idx]

        # Draw bounding box and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save image if unknown face is detected
        # if name == "Unknown":
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     filename = os.path.join(SAVE_FOLDER, f"unknown_{timestamp}.jpg")
        #     cv2.imwrite(filename, frame)
        #     print(f"Saved unknown face to {filename}")

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
