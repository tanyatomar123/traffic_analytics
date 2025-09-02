import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import sys

# Set page config
st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    page_icon="🚦",
    layout="wide"
)

# Add debug info
st.sidebar.write(f"Python version: {sys.version}")

# Try to import cv2 with fallback
try:
    import cv2
    st.sidebar.success("✅ OpenCV imported successfully!")
except ImportError:
    st.error("OpenCV not found. Installing...")
    os.system("pip install opencv-python-headless==4.8.1.78")
    try:
        import cv2
        st.sidebar.success("✅ OpenCV installed and imported!")
    except ImportError:
        st.error("Failed to install OpenCV. Please check dependencies.")
        st.stop()

# Try importing Ultralytics
try:
    from ultralytics import YOLO
    st.sidebar.success("✅ Ultralytics imported successfully!")
except ImportError:
    st.error("Ultralytics not found. Installing compatible version...")
    os.system("pip install ultralytics==8.0.196")
    try:
        from ultralytics import YOLO
        st.sidebar.success("✅ Ultralytics installed and imported!")
    except ImportError:
        st.error("Failed to install Ultralytics. Please check dependencies.")
        st.stop()

# App title and description
st.title("🚦 Traffic Analytics Dashboard")
st.markdown("""
This app uses YOLOv8 for vehicle detection and traffic analysis. Upload a video or image to get started.
""")

# Sidebar
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
    index=0
)

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model(model_type)

if model is None:
    st.warning("Could not load selected model. Using default model...")
    model = load_model("yolov8n.pt")

# File upload
uploaded_file = st.file_uploader(
    "Upload video or image",
    type=["mp4", "avi", "mov", "jpg", "jpeg", "png"],
    help="Supported formats: MP4, AVI, MOV, JPG, PNG"
)

def process_image(image, model, conf_threshold):
    """Process single image"""
    try:
        results = model(image, conf=conf_threshold)
        result = results[0]
        
        # Draw bounding boxes
        annotated_image = result.plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Count vehicles
        vehicle_count = len(result.boxes)
        
        return annotated_image_rgb, vehicle_count
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return image, 0

def process_video(video_path, model, conf_threshold):
    """Process video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open video file")
            return None, []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        st.info(f"Video info: {total_frames} frames, {fps} FPS")
        
        # Create temporary output file
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = temp_output.name
        temp_output.close()
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_count = 0
        vehicle_counts = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            results = model(frame, conf=conf_threshold)
            result = results[0]
            
            # Draw bounding boxes
            annotated_frame = result.plot()
            
            # Initialize video writer
            if out is None:
                height, width = annotated_frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            out.write(annotated_frame)
            
            # Count vehicles
            vehicle_count = len(result.boxes)
            vehicle_counts.append(vehicle_count)
            
            frame_count += 1
            if total_frames > 0:
                progress = frame_count / total_frames
                progress_bar.progress(min(progress, 1.0))
            status_text.text(f"Processing frame {frame_count} - Vehicles: {vehicle_count}")
        
        cap.release()
        if out is not None:
            out.release()
        
        progress_bar.empty()
        status_text.empty()
        
        return output_path, vehicle_counts
        
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None, []

if uploaded_file is not None:
    # Determine file type
    file_type = uploaded_file.type.split('/')[0]
    
    if file_type == 'image':
        # Process image
        try:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Processed Image")
                with st.spinner("Processing image..."):
                    processed_image, vehicle_count = process_image(image_np, model, confidence_threshold)
                    st.image(processed_image, use_column_width=True)
                    st.success(f"Detected {vehicle_count} vehicles")
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    else:
        # Process video
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.read())
                video_path = temp_file.name
            
            st.subheader("Original Video")
            st.video(uploaded_file)
            
            st.subheader("Processing Video...")
            output_path, vehicle_counts = process_video(video_path, model, confidence_threshold)
            
            if output_path and os.path.exists(output_path):
                st.subheader("Processed Video")
                st.video(output_path)
                
                # Display statistics
                if vehicle_counts:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Frames", len(vehicle_counts))
                    with col2:
                        st.metric("Max Vehicles", max(vehicle_counts))
                    with col3:
                        st.metric("Avg Vehicles", f"{np.mean(vehicle_counts):.1f}")
                
                # Cleanup
                os.unlink(output_path)
            os.unlink(video_path)
            
        except Exception as e:
            st.error(f"Error processing video: {e}")

else:
    # Demo section
    st.info("👆 Upload a video or image to start analysis")
    
    # Sample usage instructions
    st.markdown("""
    ### How to use:
    1. Upload a traffic video or image
    2. Adjust confidence threshold if needed
    3. View the processed results with vehicle detection
    4. Analyze traffic patterns and vehicle counts
    
    ### Supported vehicles:
    - Cars
    - Trucks
    - Buses
    - Motorcycles
    - and more...
    """)

# Footer
st.markdown("---")
st.caption("Traffic Analytics Dashboard | Built with YOLOv8 and Streamlit")

# Add debug info in sidebar
if st.sidebar.checkbox("Show debug info"):
    st.sidebar.write("### Debug Information")
    st.sidebar.write(f"Model: {model_type}")
    st.sidebar.write(f"Confidence threshold: {confidence_threshold}")
    st.sidebar.write("All libraries loaded successfully!")
