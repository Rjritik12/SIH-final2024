from flask import Flask, render_template, Response, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import winsound
import os
from PIL import Image
import threading

app = Flask(__name__)

# Global variables
camera = None
crowd_threshold = 2
criminal_database = {}
missing_person_data = None
missing_object_image = None
alert_active = False

# Load criminal database
def load_criminal_database():
    criminal_folder = "criminal_database"
    if os.path.exists(criminal_folder):
        for filename in os.listdir(criminal_folder):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                name = filename.split('.')[0]
                img_path = os.path.join(criminal_folder, filename)
                criminal_database[name] = img_path

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

            if len(good_matches) > 10:
                generate_alert()
                cv2.putText(frame, "Object Found!", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def process_missing_person(frame):
    global missing_person_data
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
                    cv2.putText(frame, "Missing Person Found!", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
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
            except Exception as e:
                print(f"Error in verifying criminal: {e}")
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
    file = request.files['file']
    img = Image.open(file.stream)
    missing_object_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return jsonify({"status": "success"})

@app.route('/upload_missing_person', methods=['POST'])
def upload_missing_person():
    global missing_person_data
    file = request.files['file']
    img = Image.open(file.stream)
    missing_person_data = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return jsonify({"status": "success"})

@app.route('/upload_criminal', methods=['POST'])
def upload_criminal():
    file = request.files['file']
    name = request.form['name']

    file_extension = file.filename.split('.')[-1]
    valid_extensions = ['jpg', 'jpeg', 'png']

    if file_extension.lower() not in valid_extensions:
        return jsonify({"status": "error", "message": "Invalid file extension"}), 400

    img_path = f"criminal_database/{name}.{file_extension}"
    img = Image.open(file.stream)
    img.save(img_path)
    criminal_database[name] = img_path
    return jsonify({"status": "success"})

if __name__ == '__main__':
    os.makedirs("criminal_database", exist_ok=True)
    load_criminal_database()
    app.run(debug=True)
