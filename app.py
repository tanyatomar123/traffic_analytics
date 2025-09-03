# streamlit_app.py
import streamlit as st
import torch
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO
import cv2
import tempfile
import os
import time

class BaggageDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Load the specified model
        self.model = YOLO(model_path)
        
        # Expanded baggage-related classes from COCO dataset
        self.baggage_classes = {
            24: 'backpack',
            26: 'handbag', 
            28: 'suitcase',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            67: 'cell phone',
            73: 'book',
        }
        
        self.baggage_keywords = ['bag', 'case', 'pack', 'bottle', 'container', 'box', 'suitcase', 'backpack', 'handbag']

    def detect_image(self, image):
        """Detect baggage in a single image"""
        if image is None:
            return None, "Please upload an image"
        
        try:
            # Convert to numpy array
            if isinstance(image, Image.Image):
                image_np = np.array(image)
                original_image = image.copy()
            else:
                image_np = image
                original_image = Image.fromarray(image)
            
            # Run inference
            results = self.model(image_np, conf=0.25, verbose=False)
            
            # Process results
            detections = []
            annotated_image = original_image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for box, confidence, class_id in zip(boxes, confidences, class_ids):
                        class_name = self.model.names.get(class_id, f"class_{class_id}")
                        
                        # Check if it's baggage-related
                        is_baggage = (
                            class_id in self.baggage_classes or
                            any(keyword in class_name.lower() for keyword in self.baggage_keywords)
                        )
                        
                        if is_baggage and confidence > 0.2:
                            x1, y1, x2, y2 = map(int, box)
                            
                            detections.append({
                                'class': class_name,
                                'confidence': float(confidence),
                                'bbox': [x1, y1, x2, y2]
                            })
                            
                            # Draw bounding box
                            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                            
                            # Draw label
                            label = f"{class_name}: {confidence:.2f}"
                            text_width = len(label) * 8
                            draw.rectangle([x1, y1-25, x1 + text_width, y1], fill="red")
                            draw.text((x1, y1-25), label, fill="white")
            
            return annotated_image, detections
            
        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    def detect_video(self, video_path, progress_bar=None):
        """Detect baggage in a video and create annotated output"""
        if video_path is None:
            return None, "Please upload a video"
        
        try:
            # Create temporary output file
            output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None, "Error opening video file"
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process full video
            max_frames = total_frames
            
            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            total_detections = 0
            
            # Progress bar
            if progress_bar:
                progress_bar.progress(0)
            
            # Process video frame by frame
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect baggage in frame
                results = self.model(frame, conf=0.25, verbose=False)
                
                frame_detections = 0
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for box, confidence, class_id in zip(boxes, confidences, class_ids):
                            class_name = self.model.names.get(class_id, f"class_{class_id}")
                            
                            # Check if it's baggage-related
                            is_baggage = (
                                class_id in self.baggage_classes or
                                any(keyword in class_name.lower() for keyword in self.baggage_keywords)
                            )
                            
                            if is_baggage and confidence > 0.2:
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                
                                # Draw label
                                label = f"{class_name}: {confidence:.2f}"
                                cv2.rectangle(frame, (x1, y1-30), (x1 + len(label)*10, y1), (0, 0, 255), -1)
                                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                
                                frame_detections += 1
                                total_detections += 1
                
                # Write frame to output video
                out.write(frame)
                frame_count += 1
                
                # Update progress
                if progress_bar and total_frames > 0:
                    progress_bar.progress(frame_count / total_frames)
            
            # Release resources
            cap.release()
            out.release()
            
            # Generate summary
            if total_detections > 0:
                summary = f"🎥 VIDEO ANALYSIS COMPLETE!\n\n"
                summary += f"📊 Statistics:\n"
                summary += f"• Frames processed: {frame_count}/{total_frames}\n"
                summary += f"• Total baggage detections: {total_detections}\n"
                summary += f"• Average detections per frame: {total_detections/max(1, frame_count):.2f}\n\n"
                summary += f"✅ Detection successful! Download the annotated video below."
            else:
                summary = f"❎ NO BAGGAGE DETECTED IN VIDEO\n\n"
                summary += f"Processed {frame_count} frames\n"
                summary += f"Try a video with clearer baggage items"
            
            return output_path, summary
            
        except Exception as e:
            return None, f"Error processing video: {str(e)}"

def main():
    st.set_page_config(
        page_title="Baggage Detection System",
        page_icon="🧳",
        layout="wide"
    )
    
    st.title("🧳 BAGGAGE DETECTION SYSTEM")
    st.markdown("### Supports both Images and Videos with full video processing!")
    
    # Model selection
    st.sidebar.header("Model Configuration")
    model_option = st.sidebar.selectbox(
        "Choose YOLO Model",
        ["YOLOv8 Nano (fast)", "YOLOv8 Small", "YOLOv8 Medium", "YOLOv8 Large", "Custom Model"]
    )
    
    custom_model_path = None
    if model_option == "Custom Model":
        custom_model_file = st.sidebar.file_uploader("Upload custom YOLO model", type=["pt"])
        if custom_model_file:
            # Save uploaded model temporarily
            custom_model_path = f"temp_model_{int(time.time())}.pt"
            with open(custom_model_path, "wb") as f:
                f.write(custom_model_file.getbuffer())
            st.sidebar.success("Custom model uploaded!")
    
    # Map model selection to actual model paths
    model_paths = {
        "YOLOv8 Nano (fast)": "yolov8n.pt",
        "YOLOv8 Small": "yolov8s.pt",
        "YOLOv8 Medium": "yolov8m.pt",
        "YOLOv8 Large": "yolov8l.pt"
    }
    
    # Initialize detector
    if model_option == "Custom Model" and custom_model_path:
        detector = BaggageDetector(custom_model_path)
    else:
        detector = BaggageDetector(model_paths[model_option])
    
    # Create tabs
    tab1, tab2 = st.tabs(["📷 Image Detection", "🎥 Video Detection"])
    
    with tab1:
        st.header("Image Baggage Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_image = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png"],
                key="image_upload"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Original Image", use_column_width=True)
                
                if st.button("🔍 Detect Baggage in Image", type="primary"):
                    with st.spinner("Processing image..."):
                        annotated_img, detections = detector.detect_image(image)
                        
                        if annotated_img:
                            with col2:
                                st.image(annotated_img, caption="Detected Baggage", use_column_width=True)
                                
                                if detections:
                                    st.success(f"✅ Found {len(detections)} baggage items!")
                                    for i, det in enumerate(detections, 1):
                                        st.write(f"{i}. **{det['class'].upper()}** (confidence: {det['confidence']:.2f})")
                                else:
                                    st.warning("❌ No baggage detected")
                                    st.info("Try: Clearer image, better lighting, or different baggage types")
    
    with tab2:
        st.header("Video Baggage Detection")
        st.info("Full video processing supported - no frame limits!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_video = st.file_uploader(
                "Upload a video",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_upload"
            )
            
            if uploaded_video:
                # Save uploaded video temporarily
                video_path = f"temp_video_{int(time.time())}.mp4"
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                # Show original video
                st.video(uploaded_video)
                
                if st.button("🔍 Detect Baggage in Video", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("Processing video..."):
                        status_text.text("Starting video processing...")
                        output_video, summary = detector.detect_video(video_path, progress_bar)
                        
                        progress_bar.empty()
                        status_text.empty()
                        
                        if output_video:
                            with col2:
                                st.success("✅ Video processing complete!")
                                
                                # Show processed video
                                with open(output_video, "rb") as video_file:
                                    video_bytes = video_file.read()
                                st.video(video_bytes)
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Processed Video",
                                    data=video_bytes,
                                    file_name="baggage_detected_video.mp4",
                                    mime="video/mp4"
                                )
                                
                                # Show summary
                                st.text_area("Analysis Summary", summary, height=150)
                                
                            # Clean up temporary files
                            try:
                                os.remove(video_path)
                                os.remove(output_video)
                                if custom_model_path and os.path.exists(custom_model_path):
                                    os.remove(custom_model_path)
                            except:
                                pass
    
    # Instructions
    st.sidebar.header("📋 Instructions")
    st.sidebar.info("""
    **Supported Formats:**
    - Images: JPG, JPEG, PNG
    - Videos: MP4, AVI, MOV, MKV
    
    **Detection Capabilities:**
    - 🎒 Backpacks & Bags
    - 💼 Handbags & Purses
    - 🧳 Suitcases & Luggage
    - 🍼 Bottles & Containers
    - 📦 Packages & Boxes
    """)
    
    st.sidebar.header("⚙️ Performance Tips")
    st.sidebar.warning("""
    - Larger models (Large, Medium) are more accurate but slower
    - Nano model is fastest but less accurate
    - Video processing time depends on length and model size
    """)

if __name__ == "__main__":
    main()
