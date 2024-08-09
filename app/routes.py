import random
from flask import Blueprint, Response, render_template, jsonify
import cv2
import torch
import warnings

# Suppress warnings from the torch library
warnings.filterwarnings("ignore", module="torch")

# Create a new Blueprint named 'main' for the routes
bp = Blueprint('main', __name__)

# Global variable to hold the latest object detection data
global latest_detections
latest_detections = []

# Load the YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

@bp.route('/')
def index():
    """
    Route for the index page of the application.
    Renders the 'index.html' template.
    """
    return render_template('index.html')

@bp.route('/video_feed')
def video_feed():
    """
    Route to stream the video feed from the webcam.
    Uses the 'gen' function to generate video frames and serve them as a multipart stream.
    """
    return Response(gen(model), mimetype='multipart/x-mixed-replace; boundary=frame')

@bp.route('/start_quiz')
def start_quiz():
    """
    Route to start a new quiz session.
    Generates a new question with the correct answer and two incorrect answers.
    Returns the question and answers as JSON.
    """
    question, correct_answer, incorrect_answers = generate_question()
    return jsonify({'question': question, 'correctAnswer': correct_answer, 'incorrectAnswers': incorrect_answers})

@bp.route('/next_question')
def next_question():
    """
    Route to fetch the next question in the quiz.
    Generates a new question with the correct answer and two incorrect answers.
    Returns the question and answers as JSON.
    """
    question, correct_answer, incorrect_answers = generate_question()
    return jsonify({'question': question, 'correctAnswer': correct_answer, 'incorrectAnswers': incorrect_answers})

@bp.route('/detection_data')
def detection_data():
    """
    Route to get the latest detection data from the object detection model.
    Returns the latest detections as JSON.
    """
    return jsonify(latest_detections)  # Example placeholder

@bp.route('/clear_detections', methods=['POST'])
def clear_detections():
    """
    Route to clear the latest detection data.
    Resets the global 'latest_detections' variable.
    """
    global latest_detections
    latest_detections = []  # Clear the global variable
    return jsonify({'status': 'success'})

def generate_question():
    """
    Generates a random arithmetic question with one correct answer and two incorrect answers.
    Questions can be either addition or multiplication.
    Returns the question, correct answer, and a list of two incorrect answers.
    """
    a, b = random.randint(1, 10), random.randint(1, 10)
    if random.randint(1, 10) < 6:
        question = f"{a} + {b}"
        correct_answer = a + b
    else:
        question = f"{a} × {b}"
        correct_answer = a * b

    # Generate two incorrect answers
    incorrect_answers = set()
    while len(incorrect_answers) < 2:
        wrong_answer = random.randint(1, 100)
        if wrong_answer != correct_answer:
            incorrect_answers.add(wrong_answer)

    return question, correct_answer, list(incorrect_answers)

def gen(model):
    """
    Generator function to stream video frames from the webcam.
    Performs object detection on each frame using the provided YOLOv5 model.
    Draws bounding boxes and labels on detected objects.
    Yields JPEG-encoded frames for streaming.
    """
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

        # Perform inference on the captured frame
        results = model(frame)

        # Convert detection results to numpy arrays
        labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        detected_objects = []

        for i, cord in enumerate(cords):
            label = results.names[int(labels[i])]
            x1, y1, x2, y2, conf = int(cord[0] * frame.shape[1]), int(cord[1] * frame.shape[0]), int(cord[2] * frame.shape[1]), int(cord[3] * frame.shape[0]), cord[4]
            if conf >= 0.25 and label != 'person':  # Confidence threshold
                detected_objects.append({
                    'label': label,
                    'confidence': float(conf)
                })
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Update global detection data
        latest_detections = detected_objects

        # Encode the frame as JPEG
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        # Yield the JPEG frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
