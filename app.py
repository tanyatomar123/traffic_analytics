import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial import distance as dist
from collections import deque
import os

# For object detection and tracking
import torch
from ultralytics import YOLO
from sort import Sort  # We'll implement a simple SORT tracker

# Set up the page
st.set_page_config(page_title="Traffic Analysis Dashboard", layout="wide")
st.title("🚦 Traffic Analysis Dashboard")
st.markdown("Analyze traffic camera footage to extract insights about vehicle movement and traffic patterns.")

# Add custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1f77b4; text-align: center;}
    .sub-header {font-size: 1.5rem; color: #ff7f0e; border-bottom: 2px solid #ff7f0e; padding-bottom: 0.3rem;}
    .metric-label {font-weight: bold; color: #2ca02c;}
    .violation-alert {color: #d62728; font-weight: bold;}
    .info-box {background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .lane-box {background-color: #e6f7ff; padding: 0.5rem; border-radius: 0.3rem; margin: 0.3rem 0;}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = None

# Sidebar for configuration
st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload Traffic Footage", type=["mp4", "mov", "avi", "mkv"])

# Configuration options
st.sidebar.subheader("Analysis Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
sample_interval = st.sidebar.slider("Sample Every N Frames", 1, 30, 5, 1)
speed_limit = st.sidebar.slider("Speed Limit (km/h)", 10, 120, 60, 5)

# Lane configuration
st.sidebar.subheader("Lane Configuration")
num_lanes = st.sidebar.slider("Number of Lanes", 1, 6, 3, 1)

# Function to load the pre-trained model
@st.cache_resource
def load_model():
    try:
        # Try to load a pre-trained YOLOv8 model (small version for speed)
        model = YOLO('yolov8s.pt')
        return model
    except:
        st.error("Could not load YOLO model. Please check if the model file exists.")
        return None

# Simple SORT tracker implementation
class SimpleTracker:
    def __init__(self, max_age=5):
        self.next_id = 0
        self.tracks = {}
        self.max_age = max_age
    
    def update(self, detections):
        # Simple tracking by matching with closest previous detection
        active_ids = []
        updated_tracks = {}
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Find closest existing track
            min_dist = float('inf')
            best_id = None
            
            for track_id, track in self.tracks.items():
                t_cx, t_cy = track['position']
                d = np.sqrt((cx - t_cx)**2 + (cy - t_cy)**2)
                
                if d < min_dist and d < 50:  # Distance threshold
                    min_dist = d
                    best_id = track_id
            
            if best_id is not None:
                # Update existing track
                updated_tracks[best_id] = {
                    'position': (cx, cy),
                    'box': (x1, y1, x2, y2),
                    'class': cls,
                    'confidence': conf,
                    'age': 0
                }
                active_ids.append(best_id)
            else:
                # Create new track
                updated_tracks[self.next_id] = {
                    'position': (cx, cy),
                    'box': (x1, y1, x2, y2),
                    'class': cls,
                    'confidence': conf,
                    'age': 0
                }
                active_ids.append(self.next_id)
                self.next_id += 1
        
        # Update ages and remove old tracks
        for track_id, track in self.tracks.items():
            if track_id not in active_ids:
                track['age'] += 1
                if track['age'] <= self.max_age:
                    updated_tracks[track_id] = track
        
        self.tracks = updated_tracks
        return self.tracks

# Function to process video and extract insights
def process_video(video_path, model, num_lanes=3, sample_interval=5, conf_threshold=0.5, speed_limit=60):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Define lanes (dividing frame vertically)
    lane_width = width // num_lanes
    lanes = [i * lane_width for i in range(num_lanes + 1)]
    
    # Initialize data structures for tracking
    lane_counts = defaultdict(int)
    vehicle_speeds = []
    timestamps = []
    frame_counts = []
    violations = []
    hourly_counts = defaultdict(int)
    
    # Initialize tracker
    tracker = SimpleTracker(max_age=10)
    
    # For tracking vehicles between frames
    prev_positions = {}
    speeds = {}
    
    # Process video frame by frame
    frame_idx = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a placeholder for the processed video
    video_placeholder = st.empty()
    
    # Temporary file for processed video
    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_output.close()
    
    # Video writer for processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every sample_interval-th frame to save computation
        if frame_idx % sample_interval == 0:
            status_text.text(f"Processing frame {frame_idx}/{total_frames}")
            progress_bar.progress(frame_idx / total_frames)
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect vehicles using YOLO
            results = model(frame_rgb, conf=conf_threshold, verbose=False)
            
            # Extract detections (cars, trucks, buses, motorcycles)
            vehicle_classes = [2, 3, 5, 7]  # COCO class IDs for car, motorcycle, bus, truck
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        if cls in vehicle_classes:
                            detections.append([x1, y1, x2, y2, conf, cls])
            
            # Update tracker
            tracks = tracker.update(detections)
            
            # Get current timestamp
            current_time = frame_idx / fps
            timestamps.append(current_time)
            frame_counts.append(frame_idx)
            
            # Calculate hour of day for peak traffic analysis
            hour = int((current_time % 86400) / 3600)  # Simulate time of day
            hourly_counts[hour] += len(detections)
            
            # Draw lanes
            for i in range(1, num_lanes):
                cv2.line(frame, (lanes[i], 0), (lanes[i], height), (0, 255, 255), 2)
            
            # Process each tracked vehicle
            for track_id, track in tracks.items():
                x1, y1, x2, y2 = track['box']
                cx, cy = track['position']
                cls = track['class']
                conf = track['confidence']
                
                # Draw bounding box
                label = f"ID: {track_id}"
                color = (0, 255, 0)  # Green for normal vehicles
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Determine which lane the vehicle is in
                for lane_idx in range(num_lanes):
                    if lanes[lane_idx] <= cx < lanes[lane_idx+1]:
                        lane_counts[lane_idx] += 1
                        break
                
                # Calculate speed if we have previous position
                if track_id in prev_positions:
                    prev_cx, prev_cy, prev_time = prev_positions[track_id]
                    time_diff = current_time - prev_time
                    
                    if time_diff > 0:
                        # Calculate distance traveled (pixels)
                        distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                        
                        # Convert to real-world speed (approximate)
                        # This is a simplified calculation - would need calibration IRL
                        speed_px_per_sec = distance / time_diff
                        speed_kmh = speed_px_per_sec * 0.05  # Placeholder conversion factor
                        
                        # Store speed
                        speeds[track_id] = speed_kmh
                        vehicle_speeds.append(speed_kmh)
                        
                        # Check for speeding violation
                        if speed_kmh > speed_limit:
                            violations.append({
                                'type': 'speeding',
                                'time': current_time,
                                'speed': speed_kmh,
                                'vehicle_id': track_id,
                                'lane': lane_idx
                            })
                            
                            # Draw speed warning
                            cv2.putText(frame, f"SPEEDING: {speed_kmh:.1f} km/h", 
                                       (int(x1), int(y1)-30), cv2.FONT_HERSHEY_SIMPLEX, 
                                       0.5, (0, 0, 255), 2)
                
                # Update previous position
                prev_positions[track_id] = (cx, cy, current_time)
            
            # Write frame to output video
            out.write(frame)
            
            # Display processed frame occasionally
            if frame_idx % (sample_interval * 10) == 0:
                # Resize for display
                display_frame = cv2.resize(frame, (width // 2, height // 2))
                video_placeholder.image(display_frame, channels="BGR", 
                                       caption=f"Processed Frame {frame_idx}")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Calculate insights
    avg_speed = np.mean(vehicle_speeds) if vehicle_speeds else 0
    
    # Find peak traffic hour
    peak_hour = max(hourly_counts, key=hourly_counts.get) if hourly_counts else 0
    
    # Format results
    results = {
        'lane_counts': dict(lane_counts),
        'avg_speed': avg_speed,
        'peak_hour': peak_hour,
        'hourly_counts': dict(hourly_counts),
        'timestamps': timestamps,
        'vehicle_speeds': vehicle_speeds,
        'total_vehicles': sum(lane_counts.values()),
        'violations': violations,
        'processed_video_path': temp_output.name,
        'fps': fps,
        'duration': duration
    }
    
    return results

# Function to display results
def display_results(results):
    st.success("Analysis completed successfully!")
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Vehicles", results['total_vehicles'])
    with col2:
        st.metric("Average Speed", f"{results['avg_speed']:.1f} km/h")
    with col3:
        st.metric("Peak Hour", f"{results['peak_hour']}:00")
    with col4:
        st.metric("Violations", len(results['violations']))
    
    # Display processed video
    st.subheader("Processed Video with Detections")
    st.video(results['processed_video_path'])
    
    # Display insights in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Lane Analysis", "Speed Analysis", "Time Analysis", "Violations"])
    
    with tab1:
        st.subheader("Vehicle Distribution by Lane")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            for lane, count in results['lane_counts'].items():
                st.markdown(f'<div class="lane-box">Lane {lane+1}: {count} vehicles</div>', 
                           unsafe_allow_html=True)
        
        with col2:
            fig = px.bar(x=[f"Lane {i+1}" for i in results['lane_counts'].keys()], 
                        y=list(results['lane_counts'].values()),
                        labels={'x': 'Lane', 'y': 'Number of Vehicles'},
                        title="Vehicles per Lane")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Speed Analysis")
        
        if results['vehicle_speeds']:
            fig = px.histogram(x=results['vehicle_speeds'], 
                              labels={'x': 'Speed (km/h)', 'y': 'Count'},
                              title="Speed Distribution of Vehicles")
            st.plotly_chart(fig, use_container_width=True)
            
            # Speed statistics
            speed_stats = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Value': [
                    np.mean(results['vehicle_speeds']),
                    np.median(results['vehicle_speeds']),
                    np.std(results['vehicle_speeds']),
                    np.min(results['vehicle_speeds']),
                    np.max(results['vehicle_speeds'])
                ]
            })
            st.dataframe(speed_stats, hide_index=True)
        else:
            st.info("No speed data available")
    
    with tab3:
        st.subheader("Time Analysis")
        
        # Create hourly distribution data
        hours = list(range(24))
        counts = [results['hourly_counts'].get(h, 0) for h in hours]
        
        fig = px.line(x=hours, y=counts, 
                     labels={'x': 'Hour of Day', 'y': 'Number of Vehicles'},
                     title="Traffic Volume by Hour")
        fig.update_xaxes(tickvals=list(range(0, 24, 2)))
        st.plotly_chart(fig, use_container_width=True)
        
        st.write(f"**Peak traffic hour:** {results['peak_hour']}:00")
    
    with tab4:
        st.subheader("Traffic Violations")
        
        if results['violations']:
            violations_df = pd.DataFrame(results['violations'])
            st.dataframe(violations_df)
            
            # Count violations by type
            violation_counts = violations_df['type'].value_counts()
            fig = px.pie(values=violation_counts.values, 
                        names=violation_counts.index,
                        title="Violation Types")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No traffic violations detected!")

# Main application logic
def main():
    model = load_model()
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.session_state.video_path = tfile.name
        
        # Display uploaded video
        st.header("Uploaded Traffic Footage")
        st.video(st.session_state.video_path)
        
        # Process video when button is clicked
        if st.sidebar.button("Analyze Traffic") or st.session_state.processing:
            st.session_state.processing = True
            with st.spinner("Processing video... This may take a few minutes."):
                results = process_video(
                    st.session_state.video_path, 
                    model, 
                    num_lanes=num_lanes,
                    sample_interval=sample_interval,
                    conf_threshold=confidence_threshold,
                    speed_limit=speed_limit
                )
            
            if results:
                st.session_state.results = results
                st.session_state.analysis_done = True
                st.session_state.processing = False
        
        # Display results if analysis is done
        if st.session_state.analysis_done:
            display_results(st.session_state.results)
    else:
        st.info("👈 Please upload a traffic camera footage video to begin analysis.")
        
        # Show sample video or demo
        st.subheader("How it works:")
        st.markdown("""
        1. Upload a traffic camera footage video
        2. Configure the analysis settings in the sidebar
        3. Click 'Analyze Traffic' to process the video
        4. View the results including:
           - Vehicle counts per lane
           - Average speed of vehicles
           - Peak traffic times
           - Traffic violations
        """)
        
        # Display a sample image or demo
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://i.imgur.com/7Vc6fz3.png", caption="Vehicle Detection Example")
        with col2:
            st.image("https://i.imgur.com/5tG3T9x.png", caption="Traffic Analysis Dashboard")

if __name__ == "__main__":
    main()
