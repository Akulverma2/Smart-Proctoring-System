import cv2
import pandas as pd
import os
import time
from datetime import datetime
import winsound
from ultralytics import YOLO
# Setup folders
os.makedirs("violations", exist_ok=True)
# Load models
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

model = YOLO("yolov8n.pt")  # lightweight YOLO
# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Video recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("recording.avi", fourcc, 20.0, (640, 480))
violations = 0
last_alert = 0
cooldown = 2
frame_count = 0
def log_violation(status, frame):
    global violations, last_alert

    if time.time() - last_alert < cooldown:
        return

    last_alert = time.time()
    violations += 1

    winsound.Beep(1000, 300)

    timestamp = datetime.now()
    filename = timestamp.strftime("violations/%Y%m%d_%H%M%S.jpg")
    cv2.imwrite(filename, frame)

    log_data = {
        "Time": timestamp,
        "Violation": status,
        "File": filename
    }

    pd.DataFrame([log_data]).to_csv(
        "log.csv", mode='a',
        header=not os.path.exists("log.csv"),
        index=False
    )
while True:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    status = "Normal"
    # Face + eye detection
    if len(faces) == 0:
        status = "No Face"
        log_violation(status, frame)

    elif len(faces) > 1:
        status = "Multiple Faces"
        log_violation(status, frame)

    else:
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi)

        if len(eyes) == 0:
            status = "Looking Away"
            log_violation(status, frame)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # YOLO phone detection (optimized)
    frame_count += 1
    if frame_count % 10 == 0:
        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label == "cell phone":
                    status = "Phone Detected"
                    log_violation(status, frame)
    # Head movement detection
    if len(faces) == 1:
        x, y, w, h = faces[0]
        center_x = x + w // 2

        if center_x < 200:
            status = "Looking Left"
            log_violation(status, frame)
        elif center_x > 440:
            status = "Looking Right"
            log_violation(status, frame)
    # UI DESIGN
    color = (0, 255, 0) if status == "Normal" else (0, 0, 255)

    # Top bar
    cv2.rectangle(frame, (0, 0), (640, 80), (0, 0, 0), -1)

    cv2.putText(frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Violations: {violations}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    time_text = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, time_text, (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    if status != "Normal":
        cv2.putText(frame, "ALERT!", (500, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    cv2.imshow("Smart Proctoring System", frame)
    cv2.moveWindow("Smart Proctoring System", 0, 0)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()