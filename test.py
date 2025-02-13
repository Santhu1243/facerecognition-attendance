import cv2
import face_recognition
import numpy as np
import os

# Define the folder where images will be stored
save_folder = r"C:\Users\santh\OneDrive\Desktop\iot\captured_images"  # Change this path
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Load known faces and their names
known_face_encodings = []
known_face_names = []

# Load images of known people
known_faces_folder = r"C:\Users\santh\OneDrive\Desktop\iot\known_faces"  # Change this path
for filename in os.listdir(known_faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(known_faces_folder, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)

        if encoding:  # Only add if encoding exists
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Use file name as person's name

# Initialize webcam
camera_index = 0  # Change this if needed
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Failed to access the camera. Try changing the camera index.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame. Exiting...")
        break

    # Convert frame to RGB (face_recognition uses RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces and their encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Display the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Face Recognition - Press "C" to Capture, "Q" to Quit', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        filename = os.path.join(save_folder, "captured_face.jpg")
        cv2.imwrite(filename, frame)
        print(f"📷 Image saved at: {filename}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
