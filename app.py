import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

# Set page config
st.set_page_config(
    page_title="Baggage Detection App",
    page_icon="🧳",
    layout="wide"
)

# Title and description
st.title("🧳 Baggage Detection with YOLOv8")
st.markdown("Upload a video to detect baggage using your trained YOLOv8 model")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.3, 0.1)
    st.markdown("---")
    st.info("This app uses your trained YOLOv8 model to detect baggage in videos")

# Load model function with caching
@st.cache_resource
def load_model():
    try:
        model = YOLO("baggage_detection/yolov8_baggage2/weights/best.pt")
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload a video file", 
    type=['mp4', 'mov', 'avi', 'mkv', 'MOV'],
    help="Supported formats: MP4, MOV, AVI, MKV"
)

if uploaded_file is not None and model is not None:
    # Save uploaded file to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mov') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Display original video
    st.subheader("📹 Original Video")
    st.video(video_path)

    # Process video button
    if st.button("🚀 Process Video", type="primary"):
        with st.spinner("Processing video... This may take a few minutes depending on video length"):
            
            # Create output video path
            output_path = "detected_video.mp4"
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            processed_frames = 0
            detection_data = []
            
            # Process each frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every frame
                if frame_count % 1 == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run detection
                    results = model(frame_rgb, conf=confidence, verbose=False)
                    
                    # Get detections
                    detections = []
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                conf = float(box.conf[0])
                                class_id = int(box.cls[0])
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                detections.append({
                                    'confidence': conf,
                                    'class': class_id,
                                    'bbox': [x1, y1, x2, y2]
                                })
                    
                    # Draw bounding boxes
                    annotated_frame = results[0].plot()
                    
                    # Convert back to BGR for video writing
                    annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    out.write(annotated_frame_bgr)
                    
                    detection_data.append({
                        'frame': frame_count,
                        'detections': detections
                    })
                    
                    processed_frames += 1
                
                # Update progress
                frame_count += 1
                if frame_count % 10 == 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames} "
                                   f"({processed_frames} processed)")
            
            # Release resources
            cap.release()
            out.release()
            
            # Update progress to 100%
            progress_bar.progress(1.0)
            status_text.text("Processing complete!")
            
            # Display results
            st.success("✅ Video processing completed!")
            
            # Show detection statistics
            total_detections = sum(len(frame_data['detections']) for frame_data in detection_data)
            st.subheader("📊 Detection Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Frames", total_frames)
            with col2:
                st.metric("Processed Frames", processed_frames)
            with col3:
                st.metric("Total Detections", total_detections)
            
            # Display processed video
            st.subheader("🎯 Processed Video with Detections")
            st.video(output_path)
            
            # Download button for processed video
            with open(output_path, "rb") as file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=file,
                    file_name="baggage_detected_video.mp4",
                    mime="video/mp4"
                )
            
            # Clean up temporary files
            os.unlink(video_path)
            if os.path.exists(output_path):
                os.unlink(output_path)

elif uploaded_file is not None and model is None:
    st.error("Model failed to load. Please check if the model file exists.")

else:
    st.info("👆 Please upload a video file to get started")

# Add some information about the model
with st.expander("ℹ️ About this App"):
    st.markdown("""
    **Baggage Detection App**
    
    This Streamlit application uses a YOLOv8 model trained specifically for baggage detection.
    
    **Features:**
    - Upload videos in various formats (MP4, MOV, AVI, MKV)
    - Adjustable confidence threshold
    - Real-time processing progress
    - Download processed videos with detections
    - Detection statistics and sample frames
    
    **Model Information:**
    - Trained on custom baggage dataset
    - mAP50: 99.5% (Excellent performance!)
    - Early stopping at epoch 14
    - Optimized for baggage detection
    
    **How to use:**
    1. Upload a video file
    2. Adjust confidence threshold if needed
    3. Click 'Process Video'
    4. Wait for processing to complete
    5. View and download results
    """)

# Add footer
st.markdown("---")
st.caption("Built with 🚀 Streamlit and YOLOv8 | Baggage Detection System")
