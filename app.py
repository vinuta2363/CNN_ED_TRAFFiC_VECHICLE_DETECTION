import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO

# -------------------------------
# STREAMLIT PAGE SETUP
# -------------------------------
st.set_page_config(
    page_title="Car vs Truck Detection",
    layout="centered"
)

st.title("🚗 Car vs Truck Detection using YOLOv8")
st.write("Upload a traffic image to classify vehicles as **Car** or **Truck**")

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
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # Decode image
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Save image temporarily for YOLO
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

    IMG_AREA = img.shape[0] * img.shape[1]

    # -------------------------------
    # DRAW RESULTS
    # -------------------------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls not in [CAR, BUS, TRUCK]:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            area = w * h
            ratio = h / max(w, 1)

            # Decision logic
            if cls == TRUCK and conf > 0.4:
                label = "Truck"
            elif cls == BUS and area > 0.02 * IMG_AREA:
                label = "Truck"
            elif area > 0.03 * IMG_AREA and ratio > 0.6:
                label = "Truck"
            else:
                label = "Car"

            color = (255, 0, 0) if label == "Truck" else (0, 255, 0)

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
    # DISPLAY OUTPUT
    # -------------------------------
    st.subheader("Detection Result")
    st.image(img, use_container_width=True)
