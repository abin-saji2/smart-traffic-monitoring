import cv2
import math
from collections import defaultdict, deque
from ultralytics import YOLO

# -----------------------------
# Video selection menu
# -----------------------------
print("Select video to process:")
print("1 -> video 1")
print("2 -> video 2")
print("3 -> video 3")
print("4 -> video 4")

choice = input("Enter choice (1-4): ")

video_map = {
    "1": "videos/1.mp4",
    "2": "videos/2.mp4",
    "3": "videos/3.mp4",
    "4": "videos/4.mp4"
}

video_path = video_map.get(choice, "1.mp4")

# -----------------------------
# Model
# -----------------------------
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Cannot open video file:", video_path)
    exit()

print(f"Processing: {video_path}")

# -----------------------------
# Vehicle classes
# -----------------------------
vehicle_classes = [2, 3, 5, 7]

# -----------------------------
# Tracking memory
# -----------------------------
track_history = defaultdict(lambda: deque(maxlen=5))
counted_ids = set()
total_count = 0

count_line_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.6)

# -----------------------------
# SMOOTHING BUFFER
# -----------------------------
traffic_history = deque(maxlen=10)

# -----------------------------
# Occupancy function
# -----------------------------
def get_occupancy(boxes, frame_area):
    total_area = 0
    for (x1, y1, x2, y2) in boxes:
        total_area += (x2 - x1) * (y2 - y1)
    return total_area / frame_area if frame_area > 0 else 0

# -----------------------------
# Main loop
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=960,
        verbose=False
    )

    vehicle_count_frame = 0
    speeds = []
    boxes_list = []

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])

            if cls not in vehicle_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            track_id = int(box.id[0]) if box.id is not None else None

            vehicle_count_frame += 1
            boxes_list.append((x1, y1, x2, y2))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            if track_id is not None:
                track_history[track_id].append((cx, cy))

                if len(track_history[track_id]) >= 2:
                    pts = list(track_history[track_id])

                    dist_sum = 0
                    for i in range(1, len(pts)):
                        dist_sum += math.hypot(
                            pts[i][0] - pts[i - 1][0],
                            pts[i][1] - pts[i - 1][1]
                        )

                    speed = dist_sum / len(pts)
                    speeds.append(speed)

                if len(track_history[track_id]) >= 2:
                    (px, py), (cx2, cy2) = track_history[track_id][-2], track_history[track_id][-1]

                    if track_id not in counted_ids and py < count_line_y <= cy2:
                        total_count += 1
                        counted_ids.add(track_id)

    # -----------------------------
    # 🚦 TRAFFIC ANALYSIS (FIXED REAL VERSION)
    # -----------------------------

    frame_area = frame.shape[0] * frame.shape[1]

    occupancy = get_occupancy(boxes_list, frame_area)
    occupancy = min(occupancy * 3.0, 1.0)  # scaling correction

    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    normalized_speed = 1 / (1 + avg_speed / 10.0)

    # ترکیب واقعی‌تر (نه فقط density)
    traffic_index = (
        occupancy * 0.7 +
        normalized_speed * 0.3
    )

    traffic_history.append(traffic_index)
    stable_index = sum(traffic_history) / len(traffic_history)

    # -----------------------------
    # CLASSIFICATION (SOFT & REALISTIC)
    # -----------------------------
    if stable_index > 0.68:
        traffic_status = "Traffic Jam"
    elif stable_index > 0.39:
        traffic_status = "Heavy Traffic"
    else:
        traffic_status = "Normal Traffic"

    # -----------------------------
    # UI
    # -----------------------------
    cv2.line(frame, (0, count_line_y), (frame.shape[1], count_line_y), (255, 0, 0), 2)

    cv2.putText(frame,
                f"Vehicles: {vehicle_count_frame}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    cv2.putText(frame,
                f"Status: {traffic_status}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    cv2.imshow("Traffic Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()