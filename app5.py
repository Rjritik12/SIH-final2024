from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import winsound
import os
from PIL import Image
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Global variables
camera = None
crowd_threshold = 2
criminal_database = {}
missing_person_embedding = None
missing_object_image = None
alert_active = False
yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 model
facenet_model = load_model('facenet_keras.h5')  # Load FaceNet model

# Load criminal database
def load_criminal_database():
    criminal_folder = "criminal_database"
    if os.path.exists(criminal_folder):
        for filename in os.listdir(criminal_folder):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                name = filename.split('.')[0]
                img_path = os.path.join(criminal_folder, filename)
                criminal_database[name] = generate_embedding(img_path)

def generate_embedding(img_path):
    img = cv2.imread(img_path)
    img = preprocess_image(img)
    embedding = facenet_model.predict(img)
    return embedding

def preprocess_image(img, target_size=(160, 160)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize
    return img

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

def generate_alert():
    global alert_active
    if not alert_active:
        alert_active = True
        threading.Thread(target=lambda: winsound.Beep(2500, 1000)).start()
        threading.Timer(2.0, reset_alert).start()  # Reset alert after 2 seconds

def reset_alert():
    global alert_active
    alert_active = False

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.current_frame = None

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None
        self.current_frame = frame
        return frame

def process_crowd_monitoring(frame):
    global yolo_model
    try:
        results = yolo_model(frame)  # Run inference on the frame

        # Ensure results are processed correctly
        person_count = 0
        for result in results:
            boxes = result.boxes  # Extract the bounding boxes
            for box in boxes:
                class_id = int(box.cls)  # Get the class ID
                if class_id == 0:  # Assuming class ID 0 corresponds to "person"
                    person_count += 1

        if person_count > crowd_threshold:
            generate_alert()

        cv2.putText(frame, f"People Count: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except Exception as e:
        print(f"Error in crowd monitoring: {e}")
        cv2.putText(frame, "Error in YOLO Model", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return frame

def process_missing_object(frame):
    global missing_object_image
    if missing_object_image is not None:
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(missing_object_image, None)
        kp2, des2 = sift.detectAndCompute(frame, None)

        if des1 is not None and des2 is not None:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:  # Minimum number of matches for detection
                # Extract matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Find Homography
                matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if matrix is not None:
                    h, w = missing_object_image.shape[:2]
                    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, matrix)

                    # Draw bounding box on the frame
                    frame = cv2.polylines(frame, [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=3)

                # Display "Object Found" message
                generate_alert()
                cv2.putText(frame, "Object Found!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

def process_missing_person(frame):
    global missing_person_embedding
    if missing_person_embedding is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_frame = frame[y:y+h, x:x+w]
            face_frame = preprocess_image(face_frame)
            embedding = facenet_model.predict(face_frame)

            similarity = calculate_similarity(missing_person_embedding, embedding)
            if similarity > 0.8:  # Threshold for cosine similarity
                generate_alert()
                cv2.putText(frame, "Missing Person Found!", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame

def process_criminal_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        face_frame = preprocess_image(face_frame)
        embedding = facenet_model.predict(face_frame)

        for criminal_name, criminal_embedding in criminal_database.items():
            similarity = calculate_similarity(criminal_embedding, embedding)
            if similarity > 0.8:  # Threshold for cosine similarity
                generate_alert()
                cv2.putText(frame, f"Criminal Detected: {criminal_name}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame

def gen_frames(mode):
    global camera
    if camera is None:
        camera = VideoCamera()

    while True:
        frame = camera.get_frame()
        if frame is None:
            continue

        if mode == "crowd":
            frame = process_crowd_monitoring(frame)
        elif mode == "object":
            frame = process_missing_object(frame)
        elif mode == "person":
            frame = process_missing_person(frame)
        elif mode == "criminal":
            frame = process_criminal_detection(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/<mode>')
def video_feed(mode):
    return Response(gen_frames(mode),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_missing_object', methods=['POST'])
def upload_missing_object():
    global missing_object_image
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    missing_object_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return "Missing object uploaded successfully!"

@app.route('/upload_missing_person', methods=['POST'])
def upload_missing_person():
    global missing_person_embedding
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = preprocess_image(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    missing_person_embedding = facenet_model.predict(img)
    return "Missing person uploaded successfully!"

if __name__ == "__main__":
    load_criminal_database()
    app.run(debug=True)
