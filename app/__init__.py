import os
from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    if test_config is None:
        # Load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # Load the test config if passed in
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    selected_classes = []

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video_feed')
    def video_feed():
        return Response(gen(model, selected_classes), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/select_classes', methods=['POST'])
    def select_classes():
        nonlocal selected_classes
        data = request.get_json()
        selected_classes = data.get('classes', [])
        return jsonify({'status': 'success', 'selected_classes': selected_classes})

    def gen(model, selected_classes):
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            # Perform inference
            results = model(frame)

            # Filter results based on selected classes
            labels, cords = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
            for i, cord in enumerate(cords):
                label = results.names[int(labels[i])]
                if label not in selected_classes:
                    continue
                
                x1, y1, x2, y2, conf = int(cord[0] * frame.shape[1]), int(cord[1] * frame.shape[0]), int(cord[2] * frame.shape[1]), int(cord[3] * frame.shape[0]), cord[4]
                if conf >= 0.25:  # Confidence threshold
                    label_text = f'{label} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            
            # Encode frame as JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()

    return app
