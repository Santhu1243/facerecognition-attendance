import os
import cv2
import face_recognition
import sqlite3
import numpy as np
from datetime import datetime

# Set OpenCV backend
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Database setup
DB_PATH = "faces.db"
TABLE_NAME = "recognized_faces"

def init_db():
    """Initialize the database table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            image BLOB,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize the database
init_db()

# Define paths
KNOWN_FACES_FOLDER = "/home/chezzuser/Desktop/face_auth/facerecognition-attendance/known_faces"  # Change if needed

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

# Dictionary to track captured faces
captured_faces = {}

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Failed to access the camera.")
    exit()

print("Face recognition started. Press 'q' to exit.")

def save_face_to_db(name, image):
    """Save face image to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Convert image to binary format
    _, buffer = cv2.imencode(".jpg", image)
    image_blob = buffer.tobytes()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (name, image, timestamp) VALUES (?, ?, ?)
    """, (name, image_blob, timestamp))

    conn.commit()
    conn.close()
    print(f"✅ {name}'s face saved in database!")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_face_names[matched_idx]

            # Check if this face has already been saved in this session
            if name not in captured_faces:
                save_face_to_db(name, frame)
                captured_faces[name] = True  # Mark as saved

        # Draw bounding box and name
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
