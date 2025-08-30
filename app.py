import cv2
import numpy as np
import math
from ultralytics import YOLO
import gradio as gr
from PIL import Image

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Lane polygons
lanes = {
    "Lane 1": np.array([(698, 458),(645, 493),(705, 513),(757, 478)]),
    "Lane 2": np.array([(672, 587),(633, 639),(665, 648),(706, 597)]),
    "Lane 3": np.array([(831, 540),(738, 705),(858, 724),(900, 550)]),
    "Lane 4": np.array([(1001, 549),(1032, 708),(1147, 697),(1072, 549)]),
    "Lane 5": np.array([(1040, 435),(1231, 692),(1327, 666),(1070, 431)]),
    "Lane 6": np.array([(1292, 585),(1421, 703),(1429, 603),(1352, 543)]),
}

def point_in_lane(point, lane_polygon):
    return cv2.pointPolygonTest(lane_polygon, point, False) >= 0

def analyze_video(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)  
    pixel_per_meter = 8.0    

    total_lane_counts = {lane_name: 0 for lane_name in lanes}
    counted_vehicles = {lane_name: set() for lane_name in lanes}
    vehicle_tracks = {} 

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output.mp4", fourcc, fps, 
                          (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                track_id = int(track_id)

                if label != "car":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])   
                cx, cy = (x1 + x2) // 2, y2             

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                if track_id in vehicle_tracks:
                    prev_pos = vehicle_tracks[track_id]["pos"]
                    prev_frame = vehicle_tracks[track_id]["frame"]
                    dx, dy = cx - prev_pos[0], cy - prev_pos[1]
                    dist_pixels = math.sqrt(dx*dx + dy*dy)
                    frames_passed = cap.get(cv2.CAP_PROP_POS_FRAMES) - prev_frame

                    if frames_passed > 0:
                        speed_mps = (dist_pixels / pixel_per_meter) * (fps / frames_passed)
                        speed_kmh = speed_mps * 3.6
                        vehicle_tracks[track_id]["speeds"].append(speed_kmh)
                        avg_speed = np.mean(vehicle_tracks[track_id]["speeds"])
                        cv2.putText(frame, f"{int(avg_speed)} km/h", (x1, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                vehicle_tracks[track_id] = {
                    "pos": (cx, cy),
                    "frame": cap.get(cv2.CAP_PROP_POS_FRAMES),
                    "speeds": vehicle_tracks.get(track_id, {}).get("speeds", [])
                }

                for lane_name, polygon in lanes.items():
                    if point_in_lane((cx, cy), polygon):
                        if track_id not in counted_vehicles[lane_name]:
                            total_lane_counts[lane_name] += 1
                            counted_vehicles[lane_name].add(track_id)

        y_offset = 30
        for lane_name, count in total_lane_counts.items():
            text = f"{lane_name}: {count}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (20, y_offset - text_h - 5),
                          (20 + text_w + 10, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, (25, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            y_offset += 40

        out.write(frame)

    cap.release()
    out.release()

    return "output.mp4"

# Gradio app
demo = gr.Interface(
    fn=analyze_video,
    inputs=gr.Video(label="Upload Traffic Video"),
    outputs=gr.Video(label="Processed Video"),
    title="Traffic Analytics with YOLOv8",
    description="Upload a video to detect cars, estimate speed, and count lane-wise vehicles."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
