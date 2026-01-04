import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Vehicle Detection using YOLOv8",
    layout="centered"
)

st.title("🚗 Vehicle Detection using YOLOv8")
st.write("Upload an image to detect **Cars, Buses, and Trucks**")

# -------------------------------
# LOAD YOLO MODEL
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes = uploaded_file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Save to temp file for YOLO
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_bytes)
        img_path = tmp.name

    # -------------------------------
    # YOLO INFERENCE
    # -------------------------------
    results = model(img_path)

    # COCO class IDs
    CAR = 2
    BUS = 5
    TRUCK = 7

    # Draw detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls in [CAR, BUS]:
                label = "Car"
                color = (0, 255, 0)
            elif cls == TRUCK:
                label = "Truck"
                color = (255, 0, 0)
            else:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    # -------------------------------
    # DISPLAY RESULT
    # -------------------------------
    st.subheader("Detection Result")
    st.image(img, use_container_width=True)
