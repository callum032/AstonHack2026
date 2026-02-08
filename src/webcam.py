import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from preprocess import label_to_letter

# Load CNN model
model = load_model("models/sign_mnist_cnn.h5")

# Load MediaPipe Hand Landmarker
base_options = python.BaseOptions(
    model_asset_path="models/hand_landmarker.task"
)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
hand_detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
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

            prediction = model.predict(input_data, verbose=0)
            label = np.argmax(prediction)

            # Draw results
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)
            cv2.putText(
                frame,
                label_to_letter(label),
                (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0,255,0),
                2
            )

    cv2.imshow("Sign Language MNIST (MediaPipe)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
