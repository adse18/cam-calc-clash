# routes.py
import random
from flask import Blueprint, Response, render_template, jsonify
import cv2
import torch
import warnings
warnings.filterwarnings("ignore", module="torch")

bp = Blueprint('main', __name__)
global latest_detections
latest_detections = []

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/video_feed')
def video_feed():
    return Response(gen(model), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/start_quiz')
def start_quiz():
    question, correct_answer, incorrect_answers = generate_question()
    return jsonify({'question': question, 'correctAnswer': correct_answer, 'incorrectAnswers': incorrect_answers})

@bp.route('/next_question')
def next_question():
    question, correct_answer, incorrect_answers = generate_question()
    return jsonify({'question': question, 'correctAnswer': correct_answer, 'incorrectAnswers': incorrect_answers})

@bp.route('/detection_data')
def detection_data():
    # This function should return the latest detection data as JSON
    # Ensure you have a global or shared state for the latest detections
    return jsonify(latest_detections)  # Example placeholder

@bp.route('/clear_detections', methods=['POST'])
def clear_detections():
    global latest_detections
    latest_detections = []  # Clear the global variable
    return jsonify({'status': 'success'})

def generate_question():
    a, b = random.randint(1, 10), random.randint(1, 10)
    question = f"What is {a} + {b}?"
    correct_answer = a + b

    # Generate two incorrect answers
    incorrect_answers = set()
    while len(incorrect_answers) < 2:
        wrong_answer = random.randint(1, 20)
        if wrong_answer != correct_answer:
            incorrect_answers.add(wrong_answer)

    return question, correct_answer, list(incorrect_answers)

def gen(model):

    global latest_detections
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Unable to capture frame.")
            break

        # Perform inference
        results = model(frame)

        # Convert results to numpy arrays
        labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        detected_objects = []

        for i, cord in enumerate(cords):
            label = results.names[int(labels[i])]
            x1, y1, x2, y2, conf = int(cord[0] * frame.shape[1]), int(cord[1] * frame.shape[0]), int(cord[2] * frame.shape[1]), int(cord[3] * frame.shape[0]), cord[4]
            if conf >= 0.25 and label!='person':  # Confidence threshold
                detected_objects.append({
                    'label': label,
                    'confidence': float(conf)
                })
                # Draw bounding box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Update global detection data
        latest_detections = detected_objects

        # Encode frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Yield the JPEG frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
