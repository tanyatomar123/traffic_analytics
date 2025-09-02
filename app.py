# app.py - Streamlit Baggage Detection App
import streamlit as st
import torch
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import cv2
import tempfile
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Baggage Detection AI",
    page_icon="🧳",
    layout="wide"
)

# Initialize model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Baggage classes
BAGGAGE_CLASSES = {
    24: 'backpack',
    26: 'handbag', 
    28: 'suitcase',
    39: 'bottle',
}

def detect_image(image):
    """Detect baggage in image"""
    if image is None:
        return None, []
    
    # Convert to numpy
    image_np = np.array(image)
    
    # Run detection
    results = model(image_np, conf=0.2, verbose=False)
    
    detections = []
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if class_id in BAGGAGE_CLASSES and confidence > 0.15:
                    x1, y1, x2, y2 = map(int, box)
                    
                    detections.append({
                        'class': BAGGAGE_CLASSES[class_id],
                        'confidence': float(confidence),
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    
                    # Draw label
                    label = f"{BAGGAGE_CLASSES[class_id]}: {confidence:.2f}"
                    text_width = len(label) * 8
                    draw.rectangle([x1, y1-25, x1 + text_width, y1], fill="red")
                    draw.text((x1, y1-25), label, fill="white")
    
    return annotated_image, detections

def process_video(video_file):
    """Process video and create sample detection"""
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "Error opening video"
        
        # Process first 5 frames for demo
        frames_processed = 0
        max_frames = 5
        detections = []
        
        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Detect in this frame
            _, frame_detections = detect_image(pil_image)
            detections.extend(frame_detections)
            
            frames_processed += 1
        
        cap.release()
        os.unlink(video_path)
        
        return detections, frames_processed
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Main app
def main():
    st.title("🧳 Baggage Detection AI")
    st.markdown("Upload images or videos to detect baggage items using AI")
    
    # Sidebar
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.2)
    
    # Main content
    tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])
    
    with tab1:
        st.header("Image Detection")
        
        # Upload image
        uploaded_image = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
        
        if uploaded_image is not None:
            # Display original image
            image = Image.open(uploaded_image)
            st.image(image, caption="Original Image", use_column_width=True)
            
            # Detect
            if st.button("Detect Baggage", key="detect_img"):
                with st.spinner("Analyzing image..."):
                    result_image, detections = detect_image(image)
                    
                    if detections:
                        st.success(f"Found {len(detections)} baggage items!")
                        st.image(result_image, caption="Detection Results", use_column_width=True)
                        
                        # Show detections table
                        st.subheader("Detection Details")
                        for i, det in enumerate(detections, 1):
                            st.write(f"{i}. **{det['class'].upper()}** - Confidence: {det['confidence']:.2f}")
                    else:
                        st.warning("No baggage detected. Try a different image.")
    
    with tab2:
        st.header("Video Detection")
        st.info("Note: Video processing analyzes the first few frames for demonstration")
        
        # Upload video
        uploaded_video = st.file_uploader(
            "Choose a video...", 
            type=["mp4", "avi", "mov"],
            key="video_uploader"
        )
        
        if uploaded_video is not None:
            # Display video
            st.video(uploaded_video)
            
            # Process video
            if st.button("Analyze Video", key="analyze_video"):
                with st.spinner("Processing video frames..."):
                    detections, frames_processed = process_video(uploaded_video)
                    
                    if detections:
                        st.success(f"Found {len(detections)} baggage items in {frames_processed} frames!")
                        
                        # Show summary
                        st.subheader("Video Analysis Summary")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Frames Processed", frames_processed)
                        with col2:
                            st.metric("Total Detections", len(detections))
                        
                        # Show detections
                        st.subheader("Detected Items")
                        for i, det in enumerate(detections, 1):
                            st.write(f"{i}. **{det['class'].upper()}** - Confidence: {det['confidence']:.2f}")
                    
                    else:
                        st.warning("No baggage detected in the video.")
    
    # Instructions
    st.sidebar.header("Instructions")
    st.sidebar.info("""
    **For best results:**
    - Use clear, well-lit images
    - Ensure baggage is visible
    - Common types: suitcases, backpacks, handbags
    - Video processing analyzes first few frames
    """)
    
    st.sidebar.header("Detection Capabilities")
    st.sidebar.write("""
    - 🎒 Backpacks
    - 💼 Handbags  
    - 🧳 Suitcases
    - 🍼 Bottles
    """)

if __name__ == "__main__":
    main()
