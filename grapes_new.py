import cv2
import argparse
from ultralytics import YOLO
import numpy as np
import threading
import queue
import time
from collections import deque
import torch
import serial
import serial.tools.list_ports
from typing import Tuple, Dict, Optional


class ObjectTracker:
    def __init__(self, max_lost_frames: int = 15):
        self.tracked_objects: Dict[int, dict] = {}
        self.max_lost_frames = max_lost_frames
        self.next_id = 0
        self.class_names = {0: "Unripe", 1: "Ripe", 2: "Ripe"}

    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        xi1 = max(x1, x1_p)
        yi1 = max(y1, y1_p)
        xi2 = min(x2, x2_p)
        yi2 = min(y2, y2_p)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)

        return inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0

    def update(self, detections: list) -> list:
        for obj in self.tracked_objects.values():
            obj['lost_frames'] = obj.get('lost_frames', 0) + 1

        new_tracked = []
        used_ids = set()

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            best_match_id = None
            best_iou = 0.4

            for obj_id, obj in self.tracked_objects.items():
                if obj_id in used_ids:
                    continue
                iou = self.calculate_iou((x1, y1, x2, y2), obj['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = obj_id

            if best_match_id is not None:
                self.tracked_objects[best_match_id].update({
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'cls': cls,
                    'frames': self.tracked_objects[best_match_id]['frames'] + 1,
                    'lost_frames': 0
                })
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracked_objects[new_id] = {
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'cls': cls,
                    'frames': 1,
                    'lost_frames': 0
                }
                best_match_id = new_id

            used_ids.add(best_match_id)
            new_tracked.append((best_match_id, x1, y1, x2, y2, conf, cls))

        tracked_to_show = []
        for obj_id, obj in list(self.tracked_objects.items()):
            if obj['lost_frames'] <= self.max_lost_frames:
                tracked_to_show.append((obj_id, *obj['box'], obj['conf'], obj['cls']))
            if obj['lost_frames'] > self.max_lost_frames:
                del self.tracked_objects[obj_id]

        return tracked_to_show


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grape Ripeness Detection")
    parser.add_argument("--webcam-resolution", default=[640, 480], nargs=2, type=int)
    parser.add_argument("--confidence", default=0.5, type=float)
    parser.add_argument("--model-path", default=r"C:\Users\Lenovo\Desktop\grape_detection\grapes_ripe_unripe_model.pt", type=str)
    return parser.parse_args()


class DetectionThread(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, result_queue: queue.Queue,
                 model_path: str, confidence: float):
        super().__init__()
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.model = YOLO(r"C:\Users\Lenovo\Desktop\grape_detection\grapes_ripe_unripe_model.pt")
        self.confidence = confidence
        self.running = True
        self.daemon = True

    def run(self):
        while self.running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    results = self.model(frame, verbose=False, conf=self.confidence)[0]
                    if not self.result_queue.full():
                        self.result_queue.put(results)
            except Exception as e:
                print(f"Detection thread error: {e}")
                time.sleep(0.1)

    def stop(self):
        self.running = False


'''
def get_serial_port() -> Optional[serial.Serial]:
    COM_PORT = "COM3"
    #ports = serial.tools.list_ports.comports()
    #if ports:
    try:
        #return serial.Serial(ports[0].device, 9600, timeout=1)
        return serial.Serial(COM_PORT, 9600, timeout=1)
    except Exception as e:
        print(f"Failed to open serial port: {e}")
        return None
    #return None
'''


def get_serial_port() -> Optional[serial.Serial]:
    COM_PORT = "COM8"  # Hardcoded COM port - change this to your desired port
    while True:
        try:
            ser = serial.Serial(COM_PORT, 9600, timeout=1)
            print(f"Serial port {COM_PORT} opened successfully")
            return ser
        except Exception as e:
            print(f"Failed to open serial port {COM_PORT}: {e}")
            print("Retrying in 1 second...")
            time.sleep(1)  # Wait before retrying
            return None  # Return None if initial attempt fails, handled in main


def draw_boxes(frame: np.ndarray, tracked_objects: list, center_box: Tuple[int, int, int, int],
               detection_mode: int, tracker: ObjectTracker) -> np.ndarray:
    colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (0, 0, 255)}

    if detection_mode == 1:
        cx1, cy1, cx2, cy2 = center_box
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 255, 255), 2)
        for point in [(cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)]:
            x, y = point
            cv2.line(frame, (x - 10, y), (x + 10, y), (0, 255, 255), 2)
            cv2.line(frame, (x, y - 10), (x, y + 10), (0, 255, 255), 2)

    for obj_id, x1, y1, x2, y2, conf, cls in tracked_objects:
        if detection_mode == 1 and not ((cx1 < (x1 + x2) / 2 < cx2) and (cy1 < (y1 + y2) / 2 < cy2)):
            continue

        color = colors.get(cls, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{tracker.class_names[cls]} {conf:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        cv2.rectangle(frame, (x1, y1 - label_size[1] - baseline - 5),
                      (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        if cls == 0:  # Ripe detection (no frame count needed)
            cv2.putText(frame, "Ripe Detected", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    center_box = (frame_width // 4, frame_height // 4,
                  3 * frame_width // 4, 3 * frame_height // 4)

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    tracker = ObjectTracker()

    ser = get_serial_port()  # Initial attempt to open serial port
    if ser:
        print(f"Serial port found: {ser.port}")
    else:
        print("Continuing without serial communication initially")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to initialize camera")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    detection_thread = DetectionThread(frame_queue, result_queue,
                                       args.model_path, args.confidence)
    detection_thread.start()

    fps_deque = deque(maxlen=30)
    prev_time = time.time()
    detection_mode = 0

    print("Starting Grape Ripeness Detection...")
    print("Press 'Esc' to exit, 'Space' to toggle detection mode")

    ripe_detected_flag = False  # Flag to track if "Yes" has been printed

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue

            if frame_queue.empty():
                frame_queue.put(frame.copy())

            tracked_objects = []
            if not result_queue.empty():
                results = result_queue.get()
                detections = [(int(b.xyxy[0][0]), int(b.xyxy[0][1]),
                               int(b.xyxy[0][2]), int(b.xyxy[0][3]),
                               float(b.conf[0]), int(b.cls[0]))
                              for b in results.boxes]
                tracked_objects = tracker.update(detections)
            else:
                tracked_objects = tracker.update([])

            # Serial communication with reconnection logic
            if ser:
                if not ser.is_open:  # Check if port is still open
                    print("Serial port closed, attempting to reopen...")
                    ser.close()  # Ensure it's fully closed
                    ser = get_serial_port()  # Try to reopen

                ripe_detected = any(obj_id in tracker.tracked_objects and
                                    (tracker.tracked_objects[obj_id]['cls'] == 1 or tracker.tracked_objects[obj_id]['cls'] == 2)
                                    for obj_id, *_ in tracked_objects)

                current_state = ""
                if ripe_detected:
                    if not ripe_detected_flag:  # Only print "Yes" if it hasn't been printed before
                        current_state = "Yes"
                        ripe_detected_flag = True  # Set the flag to True
                else:
                    current_state = "No"
                    ripe_detected_flag = False  # Reset the flag when no ripe grape is detected

                print(current_state)
                try:
                    if ser.is_open:
                        ser.write(current_state.encode() + b'\n')
                        ser.flush()  # Ensure data is sent
                    else:
                        print("Serial port not open, skipping write")
                except Exception as e:
                    print(f"Serial write error: {e}")
                    ser.close()
                    ser = get_serial_port()  # Reopen on error

            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            fps_deque.append(fps)
            prev_time = current_time

            frame = draw_boxes(frame, tracked_objects, center_box, detection_mode, tracker)
            cv2.putText(frame,
                        f"FPS: {sum(fps_deque) / len(fps_deque):.1f} | Mode: {'Center' if detection_mode else 'Full'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Grape Ripeness Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Esc
                break
            elif key == 32:  # Space
                detection_mode = 1 - detection_mode

            time.sleep(0.01)  # Small delay to prevent overwhelming serial port

    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        detection_thread.stop()
        detection_thread.join(timeout=2.0)
        if ser and ser.is_open:
            ser.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
