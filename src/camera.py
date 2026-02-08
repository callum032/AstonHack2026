import cv2
from flask import Flask, Response
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from preprocess import label_to_letter

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames():
    # Load CNN model
    model = load_model("AstonHack2026/models/sign_mnist_cnn.h5")

    # Load MediaPipe Hand Landmarker
    base_options = python.BaseOptions(
        model_asset_path="AstonHack2026/models/hand_landmarker.task"
    )
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1
    )
    hand_detector = vision.HandLandmarker.create_from_options(options)

    while camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = hand_detector.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Get bounding box from landmarks
            xs = [int(lm.x * w) for lm in landmarks]
            ys = [int(lm.y * h) for lm in landmarks]

            x_min, x_max = max(min(xs)-20, 0), min(max(xs)+20, w)
            y_min, y_max = max(min(ys)-20, 0), min(max(ys)+20, h)

            hand_roi = frame[y_min:y_max, x_min:x_max]

            if hand_roi.size > 0:
                gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (28, 28))
                normalized = resized / 255.0
                input_data = normalized.reshape(1, 28, 28, 1)

                prediction = model.predict(input_data)
                predicted_label = np.argmax(prediction)
                predicted_letter = label_to_letter(predicted_label)

                cv2.putText(frame, predicted_letter, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # app.run(host='0.0.0.0', port=5001)

@app.route('/')
def index():
    return Response('index.php')

@app.route('/video')
def video_route():
    print("Accessing video stream...")

    return video()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
