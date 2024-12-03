from flask import Flask, render_template, Response, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import winsound
import os
from PIL import Image
import threading
from datetime import datetime
from geopy.distance import geodesic

app = Flask(__name__)

# Global variables
camera = None
crowd_threshold = 2
criminal_database = {}
missing_person_data = None
missing_person_name = None
missing_object_image = None
alert_active = False

# Geofence configuration
GEOFENCE_CENTER = (37.7749, -122.4194)  # Example: San Francisco coordinates
GEOFENCE_RADIUS = 5000  # 5 km
CAMERA_LOCATION = (37.7749, -122.4194)  # Static location of the camera

# Load criminal database
def load_criminal_database():
    criminal_folder = "criminal_database"
    if os.path.exists(criminal_folder):
        for filename in os.listdir(criminal_folder):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                name = filename.split('.')[0]
                img_path = os.path.join(criminal_folder, filename)
                criminal_database[name] = img_path

# Geofencing helper functions
def is_within_geofence(lat, lon):
    distance = geodesic(GEOFENCE_CENTER, (lat, lon)).meters
    return distance <= GEOFENCE_RADIUS

def log_detection(person_type, name, location):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = f"{timestamp}, {person_type}, {name}, {location[0]}, {location[1]}"
    with open("detection_log.csv", "a") as log_file:
        log_file.write(log_data + "\n")
    print(f"Logged detection: {log_data}")

def generate_alert():
    global alert_active
    if not alert_active:
        alert_active = True
        threading.Thread(target=lambda: winsound.Beep(2500, 1000)).start()
        threading.Timer(2, lambda: setattr(alert_active, False)).start()

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

# Object detection using YOLO
def process_object_detection(frame):
    if missing_object_image is not None:
        # SIFT feature matching
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
                    
            if len(good_matches) > 10:
                generate_alert()
                cv2.putText(frame, "Object Found!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def process_missing_person(frame):
    global missing_person_data, missing_person_name
    if missing_person_data is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_frame = frame[y:y+h, x:x+w]
            try:
                result = DeepFace.verify(face_frame, missing_person_data, enforce_detection=False)
                if result["verified"]:
                    generate_alert()
                    cv2.putText(frame, f"Missing Person Found: {missing_person_name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if is_within_geofence(*CAMERA_LOCATION):
                        log_detection("Missing Person", missing_person_name, CAMERA_LOCATION)
            except Exception as e:
                print(f"Error in verifying missing person: {e}")
    return frame

def process_criminal_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        for criminal_name, criminal_img in criminal_database.items():
            try:
                result = DeepFace.verify(face_frame, criminal_img, enforce_detection=False)
                if result["verified"]:
                    generate_alert()
                    cv2.putText(frame, f"Criminal Detected: {criminal_name}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    if is_within_geofence(*CAMERA_LOCATION):
                        log_detection("Criminal", criminal_name, CAMERA_LOCATION)
            except Exception as e:
                print(f"Error in verifying criminal: {e}")
    return frame

def process_crowd_monitoring(frame):
    # YOLOv8 configuration for person detection
    net = cv2.dnn.readNet("yolov8n.weights", "yolov8n.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    person_count = 0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 0 is the class ID for person
                person_count += 1
                
    if person_count > crowd_threshold:
        generate_alert()
        
    cv2.putText(frame, f"People Count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
            frame = process_object_detection(frame)
        elif mode == "person":
            frame = process_missing_person(frame)
        elif mode == "criminal":
            frame = process_criminal_detection(frame)

        ret, buffer = cv2.imencode('.jpg', frame)
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
    file = request.files['file']
    img = Image.open(file.stream)
    missing_object_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return jsonify({"status": "success"})

@app.route('/upload_missing_person', methods=['POST'])
def upload_missing_person():
    global missing_person_data, missing_person_name
    
    # Validate request data
    if 'name' not in request.form or 'file' not in request.files:
        return jsonify({"status": "error", "message": "Missing name or file"}), 400
    
    missing_person_name = request.form['name']  # Get the name of the missing person
    file = request.files['file']
    
    if not file:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    try:
        img = Image.open(file.stream)
        missing_person_data = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    load_criminal_database()
    app.run(debug=True)
