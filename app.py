from flask import Flask, request, render_template, Response, jsonify, send_from_directory
import cv2
import numpy as np
import torch
import time
import math
import os
import tempfile
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import threading
import queue
import base64
from werkzeug.utils import secure_filename
import concurrent.futures

app = Flask(__name__)

# Global variables
global_video_path = None
global_roi_points = []
global_detection_results = {
    "object_count": {"car": 0, "motorcycle": 0, "truck": 0, "pedestrian": 0, "other": 0},
    "processing_time": 0,
    "fps": 0,
    "accuracy": 0,
    "speed_warning": False
}
global_frame_queue = queue.Queue(maxsize=30)  # Increased queue size
global_processing_thread = None
global_stop_processing = False
global_processing_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Load optimized models
try:
    yolo_model = YOLO('yolov8n.pt')  # Use smaller model for faster inference
    yolo_model.conf = 0.5  # Set confidence threshold
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

try:
    # Use pretrained for now, but can be updated to use weights parameter
    rcnn_model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
    rcnn_model.eval()
    # Enable half-precision for faster inference
    if torch.cuda.is_available():
        rcnn_model = rcnn_model.cuda()
    print("R-CNN model loaded successfully")
except Exception as e:
    print(f"Error loading R-CNN model: {e}")
    rcnn_model = None

# Create upload folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload size

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    global global_video_path, global_roi_points, global_detection_results
    
    # Reset previous data
    global_roi_points = []
    global_detection_results = {
        "object_count": {"car": 0, "motorcycle": 0, "truck": 0, "pedestrian": 0, "other": 0},
        "processing_time": 0,
        "fps": 0,
        "accuracy": 0,
        "speed_warning": False
    }
    
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video selected"}), 400
    
    # Save the uploaded video
    filename = secure_filename(video_file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video_file.save(filepath)
    global_video_path = filepath
    
    # Get video info
    cap = cv2.VideoCapture(global_video_path)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video file"}), 400
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get first frame for display
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({"error": "Could not read video frame"}), 400
    
    # Encode first frame to base64 for display
    _, buffer = cv2.imencode('.jpg', frame)
    first_frame = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "success": True,
        "filename": filename,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "first_frame": first_frame
    })

@app.route('/set_roi', methods=['POST'])
def set_roi():
    global global_roi_points
    
    data = request.json
    roi_points = data.get('roi_points', [])
    
    if len(roi_points) != 4:
        return jsonify({"error": "ROI must have exactly 4 points"}), 400
    
    global_roi_points = [(int(p[0]), int(p[1])) for p in roi_points]
    
    return jsonify({"success": True, "roi_points": global_roi_points})

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global global_video_path, global_roi_points, global_processing_thread, global_stop_processing
    
    if not global_video_path:
        return jsonify({"error": "No video uploaded"}), 400
    
    if len(global_roi_points) != 4:
        return jsonify({"error": "ROI not set properly"}), 400
    
    data = request.json
    model_type = data.get('model_type', 'yolo')
    
    # Stop any existing processing
    if global_processing_thread and global_processing_thread.is_alive():
        global_stop_processing = True
        global_processing_thread.join()
    
    global_stop_processing = False
    
    # Start video processing in a separate thread
    global_processing_thread = threading.Thread(
        target=process_video_optimized,  # Use optimized version
        args=(global_video_path, global_roi_points, model_type),
        daemon=True
    )
    global_processing_thread.start()
    
    return jsonify({"success": True, "message": "Detection started"})

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            if not global_frame_queue.empty():
                frame = global_frame_queue.get()
                # Compress frame for faster streaming
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # If queue is empty, send blank frame
                time.sleep(0.1)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_results')
def get_results():
    global global_detection_results
    return jsonify(global_detection_results)

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global global_stop_processing
    global_stop_processing = True
    return jsonify({"success": True, "message": "Detection stopped"})

def optimize_detection_pipeline(frame, mask, model_type):
    # Resize frame for faster processing (maintain aspect ratio)
    scale_percent = 50  # reduce size to 50%
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    resized_frame = cv2.resize(frame, (width, height))
    resized_mask = cv2.resize(mask, (width, height))
    
    # Process the smaller frame
    if model_type == "yolo":
        detections = detect_yolo(resized_frame, resized_mask)
    else:
        detections = detect_rcnn(resized_frame, resized_mask)
    
    # Scale bounding boxes back to original size
    scale_factor = 100 / scale_percent
    for detection in detections:
        bbox = detection['bbox']
        bbox[0] = int(bbox[0] * scale_factor)
        bbox[1] = int(bbox[1] * scale_factor)
        bbox[2] = int(bbox[2] * scale_factor)
        bbox[3] = int(bbox[3] * scale_factor)
    
    return detections

def detect_yolo(frame, mask):
    # Apply ROI mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Perform detection
    results = yolo_model(masked_frame, verbose=False)  # Disable verbose output
    
    # Process results
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            cls_name = yolo_model.names[cls]
            
            # Map YOLO classes to our categories
            category = map_to_category(cls_name)
            
            if conf > 0.5:  # Confidence threshold
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'category': category,
                    'confidence': conf
                })
    
    return detections

def detect_rcnn(frame, mask):
    # Apply ROI mask
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Convert to RGB for PyTorch
    rgb_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize
    x = torchvision.transforms.ToTensor()(rgb_frame)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        x = x.cuda()
        
    # Add batch dimension
    x = x.unsqueeze(0)
    
    # Perform detection
    with torch.no_grad():
        prediction = rcnn_model(x)
    
    # Process results
    detections = []
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    # COCO dataset class names that R-CNN is trained on
    coco_names = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i]
            label = coco_names[labels[i]]
            
            # Map R-CNN classes to our categories
            category = map_to_category(label)
            
            detections.append({
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'category': category,
                'confidence': float(scores[i])
            })
    
    return detections

def map_to_category(class_name):
    # Map common detection classes to our categories
    car_classes = ['car', 'automobile', 'vehicle']
    motorcycle_classes = ['motorcycle', 'motorbike', 'bicycle']
    truck_classes = ['truck', 'bus', 'train']
    pedestrian_classes = ['person', 'pedestrian']
    
    class_name = class_name.lower()
    
    if any(c in class_name for c in car_classes):
        return 'car'
    elif any(c in class_name for c in motorcycle_classes):
        return 'motorcycle'
    elif any(c in class_name for c in truck_classes):
        return 'truck'
    elif any(c in class_name for c in pedestrian_classes):
        return 'pedestrian'
    else:
        return 'other'

def process_frame(frame, roi_points, model_type, tracked_objects, next_object_id, 
                  object_count, scale_factor, speed_estimation_factor):
    """Process a single frame - moved to separate function for parallel processing"""
    # Create copy for drawing
    display_frame = frame.copy()
    
    # Draw ROI
    cv2.polylines(display_frame, [np.array(roi_points)], True, (0, 255, 0), 2)
    
    # Create ROI mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    roi_np = np.array([roi_points], dtype=np.int32)
    cv2.fillPoly(mask, roi_np, 255)
    
    # Apply optimized detection pipeline
    detections = optimize_detection_pipeline(frame, mask, model_type)
    
    # Track objects across frames and count them
    current_detections = {}
    speed_warning = False
    
    for detection in detections:
        bbox = detection['bbox']
        category = detection['category']
        confidence = detection['confidence']
        
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Check if this detection matches any tracked object
        matched_id = None
        for obj_id, obj_data in tracked_objects.items():
            last_center = obj_data['centers'][-1]
            distance = math.sqrt((center_x - last_center[0])**2 + (center_y - last_center[1])**2)
            
            # If close enough, consider it the same object
            if distance < 50 and obj_data['category'] == category:
                matched_id = obj_id
                break
        
        # If no match, create new tracked object
        if matched_id is None:
            obj_id = next_object_id
            next_object_id += 1
            tracked_objects[obj_id] = {
                'category': category,
                'centers': [(center_x, center_y)],
                'timestamps': [time.time()],
                'counted': False
            }
            matched_id = obj_id
        else:
            # Update existing tracked object
            tracked_objects[matched_id]['centers'].append((center_x, center_y))
            tracked_objects[matched_id]['timestamps'].append(time.time())
        
        # Add to current detections
        current_detections[matched_id] = True
        
        # Calculate speed if enough positions are recorded
        speed = 0
        if len(tracked_objects[matched_id]['centers']) >= 2:
            # Calculate distance between last two points
            pts = tracked_objects[matched_id]['centers']
            times = tracked_objects[matched_id]['timestamps']
            
            # Use last 5 positions for better speed estimation, or fewer if not available
            num_points = min(5, len(pts))
            if num_points >= 2:
                dx = pts[-1][0] - pts[-num_points][0]
                dy = pts[-1][1] - pts[-num_points][1]
                distance_pixels = math.sqrt(dx*dx + dy*dy)
                
                # Convert to real-world distance using scale factor
                distance_meters = distance_pixels / scale_factor
                
                # Calculate time difference
                time_diff = times[-1] - times[-num_points]
                
                if time_diff > 0:
                    # Speed in meters per second
                    speed_mps = distance_meters / time_diff
                    
                    # Convert to km/h
                    speed = speed_mps * speed_estimation_factor
        
        # Count object if it crosses the center of ROI and hasn't been counted yet
        if not tracked_objects[matched_id]['counted']:
            # Create a polygon from ROI points
            roi_poly = np.array(roi_points)
            
            # Check if the center is inside the ROI
            point_inside = cv2.pointPolygonTest(roi_poly, (center_x, center_y), False) >= 0
            
            if point_inside:
                tracked_objects[matched_id]['counted'] = True
                object_count[category] += 1
        
        # Determine color based on speed
        color = (0, 255, 0)  # Green by default
        if speed > 30:  # If speed exceeds 30 km/h
            color = (0, 0, 255)  # Red
            speed_warning = True
        
        # Draw bounding box
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence and speed
        label = f"{category}: {confidence:.2f}, {speed:.1f} km/h"
        cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Remove objects that are no longer detected
    objects_to_remove = []
    for obj_id in tracked_objects:
        if obj_id not in current_detections:
            # Check if object was inactive for too long
            if time.time() - tracked_objects[obj_id]['timestamps'][-1] > 1.0:  # 1 second threshold
                objects_to_remove.append(obj_id)
    
    for obj_id in objects_to_remove:
        del tracked_objects[obj_id]
    
    # Draw object counts on the frame
    cv2.putText(display_frame, f"Car: {object_count['car']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Motors: {object_count['motorcycle']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"Trucks: {object_count['truck']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return display_frame, next_object_id, speed_warning

def process_video_optimized(video_path, roi_points, model_type='yolo'):
    global global_detection_results, global_frame_queue, global_stop_processing
    
    # Reset counters
    tracked_objects = {}
    object_count = {"car": 0, "motorcycle": 0, "truck": 0, "pedestrian": 0, "other": 0}
    next_object_id = 0
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video setup
    output_path = os.path.join('uploads', 'output_' + os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Scale factor for speed estimation
    scale_factor = 8*20  # 1 square foot = 8x20 pixels
    speed_estimation_factor = 3.6  # Convert m/s to km/h
    
    # Performance metrics
    processing_times = []
    start_time = time.time()
    processed_frame_count = 0
    
    # Create frame buffer for batch processing
    frame_buffer = []
    buffer_size = 4  # Process 4 frames at once for better efficiency
    
    while cap.isOpened() and not global_stop_processing:
        # Read multiple frames into buffer
        for _ in range(buffer_size):
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
        
        if not frame_buffer:
            break
        
        # Process each frame
        for frame in frame_buffer:
            frame_start_time = time.time()
            
            # Process frame
            display_frame, next_object_id, has_speed_warning = process_frame(
                frame, roi_points, model_type, tracked_objects, 
                next_object_id, object_count, scale_factor, speed_estimation_factor
            )
            
            # Calculate processing time
            frame_processing_time = time.time() - frame_start_time
            processing_times.append(frame_processing_time)
            
            # Update progress information
            processed_frame_count += 1
            progress = (processed_frame_count / total_frames) * 100
            
            # Add progress text
            cv2.putText(
                display_frame, 
                f"Progress: {progress:.1f}%", 
                (frame_width - 250, frame_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2
            )
            
            # Calculate FPS
            if processing_times:
                current_fps = 1.0 / (sum(processing_times[-30:]) / min(len(processing_times), 30))
                cv2.putText(
                    display_frame, 
                    f"FPS: {current_fps:.1f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
            
            # Write to video
            out.write(display_frame)
            
            # Put in queue for streaming (if not full)
            if not global_frame_queue.full():
                global_frame_queue.put(display_frame)
            
            # Update global results
            avg_fps = 1.0 / (sum(processing_times[-30:]) / min(len(processing_times), 30)) if processing_times else 0
            global_detection_results = {
                "object_count": object_count,
                "processing_time": time.time() - start_time,
                "fps": avg_fps,
                "progress": progress,
                "speed_warning": has_speed_warning
            }
        
        # Clear buffer for next batch
        frame_buffer = []
    
    # Release resources
    cap.release()
    out.release()
    
    # Update final results
    total_time = time.time() - start_time
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_fps = 1.0 / (avg_processing_time + 0.001)
    
    global_detection_results = {
        "object_count": object_count,
        "processing_time": total_time,
        "fps": avg_fps,
        "accuracy": "N/A",  # Would need ground truth for this
        "completed": True,
        "output_video": os.path.basename(output_path),
        "speed_warning": False
    }
    
    print("=== Detection Complete ===")
    print(f"Model: {model_type.upper()}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average frame processing time: {avg_processing_time*1000:.2f} ms")
    print("Final counts:")
    print(f"Cars: {object_count['car']}")
    print(f"Motorcycles: {object_count['motorcycle']}")
    print(f"Trucks: {object_count['truck']}")
    print(f"Pedestrians: {object_count['pedestrian']}")
    print(f"Other: {object_count['other']}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)