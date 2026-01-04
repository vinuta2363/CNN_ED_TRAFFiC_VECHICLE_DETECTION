import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

st.set_page_config(page_title="Vehicle Detection", layout="centered")
st.title("🚗 Vehicle Detection using YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", ["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = uploaded_file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    results = model(temp_path)

    CAR, BUS, TRUCK = 2, 5, 7

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == CAR or cls == BUS:
                label, color = "Car", (0, 255, 0)
            elif cls == TRUCK:
                label, color = "Truck", (255, 0, 0)
            else:
                continue

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    st.subheader("Detection Result")
    st.image(img, use_container_width=True)
