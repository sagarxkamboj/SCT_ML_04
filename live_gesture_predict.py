import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model("gesture_cnn_model.h5")

# Define label map (must match folder names used during training)
label_map = {
    0: "01_palm",
    1: "02_l",
    2: "03_fist",
    3: "04_fist_moved",
    4: "05_thumb",
    5: "06_index",
    6: "07_ok",
    7: "08_palm_moved",
    8: "09_c",
    9: "10_down"
}


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("📷 Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image (optional)
    frame = cv2.flip(frame, 1)

    # Define region of interest (ROI)
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = frame[y1:y2, x1:x2]

    # Preprocess the ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    normalized = resized.astype("float32") / 255.0
    reshaped = normalized.reshape(1, 64, 64, 1)

    # Predict gesture
    predictions = model.predict(reshaped)
    predicted_class = np.argmax(predictions)
    gesture_name = label_map[predicted_class]

    # Draw ROI box and prediction
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture_name}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Live Gesture Prediction", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
