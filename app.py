import streamlit as st
from ultralytics import YOLO
import tempfile
import os
import cv2

st.set_page_config(page_title="Baggage Detection (YOLOv8)", layout="wide")
st.title("🎒 Baggage Detection (YOLOv8)")

# --- MODEL UPLOAD ---
st.sidebar.header("Model Upload")
model_file = st.sidebar.file_uploader("Upload YOLOv8 model (.pt)", type=["pt"])

model = None
if model_file:
    # Save model to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name
    try:
        model = YOLO(model_path)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")

# --- IMAGE / VIDEO UPLOAD ---
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload an Image or Video", 
                                         type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if model and uploaded_file:
    # Save uploaded input
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(uploaded_file.read())
        input_path = tmp_in.name

    # Run YOLOv8
    with st.spinner("Running detection..."):
        results = model.predict(source=input_path, save=True, project="runs", name="streamlit_result")

    # Find output file
    result_dir = os.path.join("runs", "streamlit_result")
    files = os.listdir(result_dir)
    if files:
        output_path = os.path.join(result_dir, files[0])
        
        # Show result inline
        if output_path.lower().endswith((".jpg", ".jpeg", ".png")):
            st.image(output_path, caption="Detection Result", use_column_width=True)
        elif output_path.lower().endswith((".mp4", ".avi", ".mov")):
            st.video(output_path)
        else:
            st.error("⚠️ Unsupported output format.")
