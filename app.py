# app.py
import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image

# --------------------------
# Load YOLO model
# --------------------------
# Use your trained model path here (e.g., "runs/detect/train/weights/best.pt")
MODEL_PATH = "yolov8n.pt"  # replace with your trained model
model = YOLO(MODEL_PATH)

st.title("🎒 Baggage Detection with YOLOv8")
st.write("Upload an image or video to detect baggage.")

# --------------------------
# Upload file
# --------------------------
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    suffix = uploaded_file.name.split(".")[-1].lower()

    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name

    if suffix in ["jpg", "jpeg", "png"]:
        # --------------------------
        # Handle Image
        # --------------------------
        st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)

        results = model(temp_file_path)  # Run YOLO inference
        for r in results:
            im_bgr = r.plot()  # YOLO result with bounding boxes
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            st.image(im_rgb, caption="Detection Result", use_column_width=True)

    else:
        # --------------------------
        # Handle Video
        # --------------------------
        st.video(temp_file_path)

        results = model.predict(source=temp_file_path, save=True, save_txt=False)

        st.success("✅ Detection complete. Processed video saved in `runs/detect/predict/` folder.")
        st.write("You can check the folder for the output video with bounding boxes.")
