#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traffic Analytics Web/Local App
- Vehicle detection (YOLOv8)
- Multi-object tracking (ByteTrack via supervision)
- Lane-wise counts
- Peak traffic time
- Avg speed
- Red-light violation detection (optional)

Usage:
    pip install -r requirements.txt
    python app.py --source path/to/video.mp4 --out result.mp4 --show
"""

import os
import cv2
import math
import time
import json
import argparse
from collections import defaultdict, deque
from datetime import datetime
import numpy as np

from ultralytics import YOLO
import supervision as sv

# ---------------- CONFIG ----------------
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}
YOLO_MODEL = "yolov8n.pt"
SPEED_WINDOW = 10
START_TIME = None  # e.g. "07:30"
# ----------------------------------------

def load_yolo(model_path=YOLO_MODEL):
    model = YOLO(model_path)
    names = model.model.names
    cls_ids = {i for i, n in names.items() if n in VEHICLE_CLASSES}
    return model, names, cls_ids

class SpeedEstimator:
    def __init__(self, fps, meters_per_pixel=0.05, window=SPEED_WINDOW):
        self.fps = fps
        self.mpp = meters_per_pixel
        self.window = window
        self.history = defaultdict(lambda: deque(maxlen=window))
        self.speeds = {}

    def update(self, track_id, cx, cy, t_now):
        h = self.history[track_id]
        h.append((t_now, cx, cy))
        if len(h) >= 2:
            t0, x0, y0 = h[0]
            t1, x1, y1 = h[-1]
            dt = max(t1 - t0, 1e-6)
            dpx = math.hypot(x1 - x0, y1 - y0)
            dm = dpx * self.mpp
            mps = dm / dt
            kmph = mps * 3.6
            self.speeds[track_id] = kmph

    def get_speed(self, track_id):
        return float(self.speeds.get(track_id, 0.0))

class LaneCounter:
    def __init__(self, lanes):
        self.lane_names = list(lanes.keys())
        self.zones = {}
        self.annotators = {}
        self.counts = {k: 0 for k in lanes.keys()}
        for name, ((x1,y1),(x2,y2)) in lanes.items():
            line = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))
            self.zones[name] = line
            self.annotators[name] = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.6)

    def update(self, detections, frame):
        for name, zone in self.zones.items():
            zone.trigger(detections=detections)
            self.counts[name] = int(zone.in_count + zone.out_count)
        for name, annot in self.annotators.items():
            frame = annot.annotate(frame=frame, line_zone=self.zones[name])
        return frame

    def get_counts(self):
        return dict(self.counts)

class ViolationDetector:
    def __init__(self, stopline=((100,100),(500,100))):
        self.stopline = sv.LineZone(start=sv.Point(*stopline[0]), end=sv.Point(*stopline[1]))
        self.annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.6)
        self.violations = 0

    def update(self, detections, frame, signal_is_red=False):
        if signal_is_red:
            before = self.stopline.in_count + self.stopline.out_count
            self.stopline.trigger(detections=detections)
            after = self.stopline.in_count + self.stopline.out_count
            self.violations += max(0, after - before)
        frame = self.annotator.annotate(frame=frame, line_zone=self.stopline)
        return frame

    def get_count(self):
        return self.violations

def draw_side_panel(frame, panel_w, lane_counts, avg_speed, peak_minute, total, violations=None):
    h, w = frame.shape[:2]
    x0 = w - panel_w
    cv2.rectangle(frame, (x0, 0), (w, h), (25, 25, 25), -1)

    y = 30
    def put(txt, scale=0.6, pad=28):
        nonlocal y
        cv2.putText(frame, txt, (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)
        y += pad

    put("Traffic Analytics", 0.8, 34)
    put(f"Total tracked: {total}")
    put(f"Avg speed: {avg_speed:.1f} km/h")
    put(f"Peak minute: {peak_minute}")
    y += 6
    put("Per-lane counts:", 0.7, 30)
    for lane, cnt in lane_counts.items():
        put(f"  {lane}: {cnt}")
    if violations is not None:
        y += 10
        put(f"Violations: {violations}")
    return frame

def minute_bucket(frame_idx, fps):
    minute = int((frame_idx / fps) // 60)
    return f"t+{minute:02d}m"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Video file or stream URL")
    ap.add_argument("--out", default="output.mp4", help="Output video file")
    ap.add_argument("--model", default=YOLO_MODEL, help="YOLO model path")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--meters_per_pixel", type=float, default=0.05, help="Scene scale (m/px)")
    ap.add_argument("--show", action="store_true", help="Show live preview")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model, names, cls_ids = load_yolo(args.model)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    lanes = {
        "Lane 1": ((int(0.15*width), int(0.65*height)), (int(0.45*width), int(0.65*height))),
        "Lane 2": ((int(0.15*width), int(0.75*height)), (int(0.45*width), int(0.75*height))),
        "Lane 3": ((int(0.15*width), int(0.85*height)), (int(0.45*width), int(0.85*height))),
    }
    lane_counter = LaneCounter(lanes)
    violation_detector = ViolationDetector()
    speed_est = SpeedEstimator(fps=fps, meters_per_pixel=args.meters_per_pixel)
    minute_hist = defaultdict(int)
    seen_ids = set()
    panel_w = max(260, int(0.28*width))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        t_now = frame_idx / fps
        results = model(frame, conf=args.conf, verbose=False)[0]
        det = sv.Detections.from_ultralytics(results)
        if det.class_id is not None and len(det) > 0:
            mask = np.array([c in cls_ids for c in det.class_id], dtype=bool)
            det = det[mask]
        det = tracker.update_with_detections(det)

        for xyxy, tid in zip(det.xyxy, det.tracker_id or []):
            if tid is None: continue
            x1,y1,x2,y2 = xyxy
            cx, cy = (x1+x2)/2, y2
            speed_est.update(int(tid), cx, cy, t_now)
            seen_ids.add(int(tid))

        frame = lane_counter.update(det, frame)
        frame = violation_detector.update(det, frame, signal_is_red=False)

        minute_hist[minute_bucket(frame_idx,fps)] += len(set(det.tracker_id or []))
        labels = []
        for i in range(len(det)):
            tid = int(det.tracker_id[i]) if det.tracker_id is not None else -1
            cname = names.get(int(det.class_id[i]), "obj")
            spd = speed_est.get_speed(tid)
            labels.append(f"#{tid} {cname} {spd:0.0f} km/h")
        frame = box_annotator.annotate(scene=frame, detections=det, labels=labels)

        lane_counts = lane_counter.get_counts()
        avg_speed = sum(speed_est.speeds.values())/max(1,len(speed_est.speeds))
        peak_minute = max(minute_hist.items(), key=lambda kv: kv[1])[0]
        frame = draw_side_panel(frame, panel_w, lane_counts, avg_speed, peak_minute,
                                len(seen_ids), violations=violation_detector.get_count())

        out.write(frame)
        if args.show:
            cv2.imshow("Traffic Analytics", frame)
            if cv2.waitKey(1) & 0xFF == 27: break
        frame_idx += 1

    cap.release(); out.release(); cv2.destroyAllWindows()
    print("=== SUMMARY ===")
    print("Lane counts:", lane_counter.get_counts())
    print("Peak minute:", max(minute_hist.items(), key=lambda kv: kv[1]))
    print("Avg speed:", sum(speed_est.speeds.values())/max(1,len(speed_est.speeds)))
    print("Violations:", violation_detector.get_count())
    print("Output saved to:", args.out)

if __name__ == "__main__":
    main()
