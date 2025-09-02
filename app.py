import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import subprocess

# Set page config
st.set_page_config(
    page_title="Traffic Analytics Dashboard",
    page_icon="🚦",
    layout="wide"
)

# Function to install missing packages
def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

# Try to import cv2 with fallback
try:
    import cv2
    st.sidebar.success("✅ OpenCV imported successfully!")
except ImportError:
    st.warning("OpenCV not found. Attempting to install...")
    if install_package("opencv-python-headless==4.5.5.64"):
        try:
            import cv2
            st.sidebar.success("✅ OpenCV installed and imported!")
        except ImportError:
            st.error("Failed to import OpenCV after installation.")
            st.stop()
    else:
        st.error("Failed to install OpenCV. Please install manually: pip install opencv-python-headless==4.5.5.64")
        st.stop()

# Try importing Ultralytics
try:
    from ultralytics import YOLO
    st.sidebar.success("✅ Ultralytics imported successfully!")
except ImportError:
    st.warning("Ultralytics not found. Attempting to install...")
    if install_package("ultralytics==8.0.20"):
        try:
            from ultralytics import YOLO
            st.sidebar.success("✅ Ultralytics installed and imported!")
        except ImportError:
            st.error("Failed to import Ultralytics after installation.")
            st.stop()
    else:
        st.error("Failed to install Ultralytics. Please install manually: pip install ultralytics==8.0.20")
        st.stop()

# App title and description
st.title("🚦 Simple Traffic Analytics Dashboard")
st.markdown("""
This app uses YOLOv8 for vehicle detection. Upload an image to get started.
""")

# Sidebar
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Use built-in YOLOv8n model (no download needed)
model_type = "yolov8n.pt"

@st.cache_resource
def load_model():
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_type)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

if model is None:
    st.error("Failed to load model. Please check your installation.")
    st.stop()

# File upload - Only images for simplicity
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, PNG"
)

def process_image(image, model, conf_threshold):
    """Process single image"""
    try:
        # Convert PIL Image to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Run inference
        results = model(image_np, conf=conf_threshold, verbose=False)
        result = results[0]
        
        # Draw bounding boxes
        annotated_image = result.plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Count vehicles (filter for vehicle classes: car, truck, bus, motorcycle)
        vehicle_classes = [2, 3, 5, 7]  # COCO dataset classes for vehicles
        vehicle_count = 0
        if result.boxes is not None:
            for box in result.boxes:
                if int(box.cls) in vehicle_classes:
                    vehicle_count += 1
        
        return annotated_image_rgb, vehicle_count
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return image, 0

if uploaded_file is not None:
    try:
        # Process image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Processed Image")
            with st.spinner("Processing image..."):
                processed_image, vehicle_count = process_image(image, model, confidence_threshold)
                st.image(processed_image, use_column_width=True)
                st.success(f"Detected {vehicle_count} vehicles")
                
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    # Demo section
    st.info("👆 Upload an image to start vehicle detection")
    
    st.markdown("""
    ### How to use:
    1. Upload a traffic image (JPG or PNG)
    2. Adjust confidence threshold if needed
    3. View the processed results with vehicle detection
    
    ### Supported vehicles:
    - Cars 🚗
    - Trucks 🚚
    - Buses 🚌
    - Motorcycles 🏍️
    """)

# Footer
st.markdown("---")
st.caption("Traffic Analytics Dashboard | Built with YOLOv8 and Streamlit")

# Debug info
if st.sidebar.checkbox("Show debug info"):
    st.sidebar.write("### System Information")
    st.sidebar.write(f"Python: {sys.version}")
    st.sidebar.write(f"OpenCV: {cv2.__version__}")
    st.sidebar.write("✅ All systems operational!")
