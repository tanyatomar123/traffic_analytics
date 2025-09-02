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

# Check and import required packages
try:
    import cv2
    st.sidebar.success("✅ OpenCV imported successfully!")
except ImportError:
    st.error("""
    ❌ OpenCV not found. Please install required packages first:
    
    **Run this command in your terminal:**
    ```bash
    pip install -r requirements.txt
    ```
    
    Then restart the app.
    """)
    st.stop()

try:
    from ultralytics import YOLO
    st.sidebar.success("✅ Ultralytics imported successfully!")
except ImportError:
    st.error("""
    ❌ Ultralytics not found. Please install required packages first:
    
    **Run this command in your terminal:**
    ```bash
    pip install -r requirements.txt
    ```
    
    Then restart the app.
    """)
    st.stop()

# App title and description
st.title("🚦 Traffic Analytics Dashboard")
st.markdown("""
This app uses YOLOv8 for vehicle detection and traffic analysis. Upload an image to get started.
""")

# Sidebar
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Use built-in YOLOv8n model
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

# File upload
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
        
        # Count vehicles
        vehicle_classes = [2, 3, 5, 7]  # COCO classes: car, motorcycle, bus, truck
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
    try:
        import ultralytics
        st.sidebar.write(f"Ultralytics: {ultralytics.__version__}")
    except:
        st.sidebar.write("Ultralytics: Not available")
    st.sidebar.write("✅ All systems operational!")
