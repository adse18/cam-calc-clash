from flask import Blueprint, Response
import cv2
import torch

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Create a blueprint
bp = Blueprint('main', __name__)

@bp.route('/video_feed')
def video_feed():
    return Response(gen(model), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen(model):
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
        
        for i, cord in enumerate(cords):
            label = results.names[int(labels[i])]
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
