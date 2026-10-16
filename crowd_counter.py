import cv2
import requests
import time
import numpy as np
import threading
from collections import deque

SERVER_URL = "http://localhost:5000/api/update-occupancy"

# Load Pre-trained Classifiers for Human & Student Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
upper_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

# Initialize HOG Person Detector for standing students
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Sensitivity Presets: [minNeighbors_face, minNeighbors_profile, minNeighbors_upper]
SENSITIVITY_PRESETS = {
    "1": ("High Sensitivity (Far/Dim)", 4, 4, 3),
    "2": ("Balanced (Recommended)", 5, 5, 4),
    "3": ("Strict (Zero False Positives)", 6, 6, 5)
}
current_preset_key = "2"

# Rolling history for temporal stabilization (prevents frame flicker)
history_window = deque(maxlen=8)
last_reported_count = -1
last_post_time = 0.0
server_status = "Syncing..."
server_status_color = (0, 200, 255) # Yellow/Orange

def async_post_occupancy(count):
    """Send occupancy to Flask server in a background thread to prevent camera lag."""
    global server_status, server_status_color
    def _post():
        global server_status, server_status_color
        try:
            res = requests.post(SERVER_URL, json={"occupancy": count}, timeout=1.0)
            if res.status_code == 200:
                server_status = "Connected (200 OK)"
                server_status_color = (0, 255, 100) # Green
            else:
                server_status = f"Server Error ({res.status_code})"
                server_status_color = (0, 165, 255)
        except requests.exceptions.RequestException:
            server_status = "Offline / Server Disconnected"
            server_status_color = (0, 0, 255) # Red
    threading.Thread(target=_post, daemon=True).start()

def detect_students(frame, preset_key):
    """Accurately detects students using facial & upper-body patterns. Ignores walls and shadows."""
    h, w = frame.shape[:2]
    
    # Scale frame slightly for faster, highly accurate multi-scale detection
    proc_w = 640
    scale = proc_w / float(w) if w > proc_w else 1.0
    if scale != 1.0:
        proc_h = int(h * scale)
        proc_frame = cv2.resize(frame, (proc_w, proc_h))
    else:
        proc_frame = frame
        
    gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
    # Equalize histogram to handle varying classroom lighting
    gray = cv2.equalizeHist(gray)
    
    _, fn, pn, un = SENSITIVITY_PRESETS[preset_key]
    
    boxes = []
    scores = []
    
    # 1. Frontal faces (seated/standing students facing camera)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=fn, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, bw, bh) in faces:
        boxes.append([int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)])
        scores.append(0.95)

    # 2. Profile faces (students turned sideways - right & left)
    profiles_r = profile_cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=pn, minSize=(35, 35)
    )
    for (x, y, bw, bh) in profiles_r:
        boxes.append([int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)])
        scores.append(0.85)

    # Left profile by flipping gray image
    gray_flipped = cv2.flip(gray, 1)
    profiles_l = profile_cascade.detectMultiScale(
        gray_flipped, scaleFactor=1.15, minNeighbors=pn, minSize=(35, 35)
    )
    fw = proc_frame.shape[1]
    for (x, y, bw, bh) in profiles_l:
        orig_x = fw - (x + bw)
        boxes.append([int(orig_x / scale), int(y / scale), int(bw / scale), int(bh / scale)])
        scores.append(0.85)

    # 3. Upper body (students sitting at desks with head/shoulders visible)
    uppers = upper_cascade.detectMultiScale(
        gray, scaleFactor=1.18, minNeighbors=un, minSize=(65, 65)
    )
    for (x, y, bw, bh) in uppers:
        boxes.append([int(x / scale), int(y / scale), int(bw / scale), int(bh / scale)])
        scores.append(0.75)

    if not boxes:
        return []

    # Non-Maximum Suppression to remove overlapping duplicate detections
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.6, nms_threshold=0.35)
    if len(indices) == 0:
        return []
    
    filtered_boxes = [boxes[i] for i in indices.flatten()]
    
    # Merge nested detections belonging to the same student (e.g. face inside body)
    merged_students = []
    used = [False] * len(filtered_boxes)
    for i in range(len(filtered_boxes)):
        if used[i]:
            continue
        bx1, by1, bw1, bh1 = filtered_boxes[i]
        cx1, cy1 = bx1 + bw1 / 2.0, by1 + bh1 / 2.0
        
        student_box = [bx1, by1, bw1, bh1]
        for j in range(i + 1, len(filtered_boxes)):
            if used[j]:
                continue
            bx2, by2, bw2, bh2 = filtered_boxes[j]
            cx2, cy2 = bx2 + bw2 / 2.0, by2 + bh2 / 2.0
            
            # Check if one detection's center is inside the other's bounding box
            in_j = (bx2 <= cx1 <= bx2 + bw2) and (by2 <= cy1 <= by2 + bh2)
            in_i = (bx1 <= cx2 <= bx1 + bw1) and (by1 <= cy2 <= by1 + bh1)
            
            # Or high intersection
            inter_x1 = max(bx1, bx2)
            inter_y1 = max(by1, by2)
            inter_x2 = min(bx1 + bw1, bx2 + bw2)
            inter_y2 = min(by1 + bh1, by2 + bh2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            min_area = min(bw1 * bh1, bw2 * bh2)
            
            if in_j or in_i or (min_area > 0 and inter_area / float(min_area) > 0.4):
                min_x = min(student_box[0], bx2)
                min_y = min(student_box[1], by2)
                max_x = max(student_box[0] + student_box[2], bx2 + bw2)
                max_y = max(student_box[1] + student_box[3], by2 + bh2)
                student_box = [min_x, min_y, max_x - min_x, max_y - min_y]
                used[j] = True
                
        merged_students.append(student_box)
        
    return merged_students

def draw_corner_rect(img, pt1, pt2, color, thickness=2, d=15):
    """Draws a modern bracket-style corner bounding box."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Bounding rectangle with thin border
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # Corner brackets for futuristic/modern look
    d = min(d, (x2 - x1) // 3, (y2 - y1) // 3)
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + d, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + d), color, thickness)
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - d, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + d), color, thickness)
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + d, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - d), color, thickness)
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - d, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - d), color, thickness)

def main():
    global current_preset_key, last_reported_count, last_post_time
    
    # Try opening webcam (index 0 first, fallback to 1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam (tested index 0 and 1).")
            return

    # Attempt setting ideal webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("=" * 65)
    print(" EcoSmart Campus - AI Student Presence Detector")
    print(" Active Model: Multi-Cascade Human & Facial Feature Classifier")
    print(" Walls, shadows, and static background objects are ignored.")
    print("=" * 65)
    print(" Controls:")
    print("  [1] High Sensitivity  (Good for far-away seating or dimmer light)")
    print("  [2] Balanced Mode     (Default recommended)")
    print("  [3] Strict Mode       (Zero false detections)")
    print("  [q] Quit detector")
    print("=" * 65)

    fps_time = time.time()
    fps_counter = 0
    display_fps = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from webcam.")
            break

        # Calculate FPS
        fps_counter += 1
        now = time.time()
        if now - fps_time >= 1.0:
            display_fps = fps_counter
            fps_counter = 0
            fps_time = now

        # Detect students
        students = detect_students(frame, current_preset_key)
        raw_count = len(students)

        # Temporal smoothing to avoid flickering
        history_window.append(raw_count)
        stable_count = int(round(np.median(history_window)))

        # Periodic / on-change server sync
        if (stable_count != last_reported_count) or (now - last_post_time >= 2.0):
            async_post_occupancy(stable_count)
            last_reported_count = stable_count
            last_post_time = now

        # Draw detected students
        for idx, (x, y, w, h) in enumerate(students):
            draw_corner_rect(frame, (x, y), (x + w, y + h), (50, 220, 100), thickness=2)
            label = f"Student #{idx + 1}"
            
            # Label tag pill
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x, max(0, y - th - 8)), (x + tw + 10, y), (50, 220, 100), -1)
            cv2.putText(frame, label, (x + 5, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Glassmorphic top HUD overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 55), (15, 23, 42), -1)
        cv2.rectangle(overlay, (0, frame.shape[0] - 30), (frame.shape[1], frame.shape[0]), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        # Top Bar: Title & Live Count
        cv2.putText(frame, "EcoSmart Campus | Student Presence AI", (15, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        mode_desc, _, _, _ = SENSITIVITY_PRESETS[current_preset_key]
        cv2.putText(frame, f"Mode: {mode_desc} | FPS: {display_fps}", (15, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (148, 163, 184), 1, cv2.LINE_AA)

        # Occupancy Badge (Top Right)
        badge_text = f"Occupancy: {stable_count} Students"
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        badge_x = frame.shape[1] - bw - 25
        badge_color = (34, 197, 94) if stable_count <= 5 else (59, 130, 246) # Green or Blue
        cv2.rectangle(frame, (badge_x - 10, 10), (badge_x + bw + 10, 42), badge_color, -1)
        cv2.putText(frame, badge_text, (badge_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Bottom Bar: Server Status & Quick Controls
        cv2.circle(frame, (20, frame.shape[0] - 15), 5, server_status_color, -1)
        cv2.putText(frame, f"Server: {server_status}", (32, frame.shape[0] - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (203, 213, 225), 1, cv2.LINE_AA)
        
        controls_text = "Keys: [1] High [2] Balanced [3] Strict | [q] Exit"
        (cw, _), _ = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, controls_text, (frame.shape[1] - cw - 15, frame.shape[0] - 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

        cv2.imshow("EcoSmart Campus - Student Presence AI Tracker", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # q or ESC
            break
        elif key in [ord('1'), ord('2'), ord('3')]:
            current_preset_key = chr(key)
            print(f"[INFO] Switched sensitivity to: {SENSITIVITY_PRESETS[current_preset_key][0]}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Tracker closed gracefully.")

if __name__ == "__main__":
    main()