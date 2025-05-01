import cv2
import numpy as np
import torch
import time
import tkinter as tk
from tkinter import filedialog, Button, Label
from PIL import Image, ImageTk
import math
import os
from ultralytics import YOLO
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import matplotlib.pyplot as plt

class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle Detection and Counting System")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.video_path = None
        self.cap = None
        self.roi_points = []
        self.roi_selected = False
        self.detection_started = False
        self.current_frame = None
        self.frame_width = 0
        self.frame_height = 0
        self.tracked_objects = {}
        self.object_count = {"car": 0, "motorcycle": 0, "truck": 0, "pedestrian": 0, "other": 0}
        self.actual_object_count = {"car": 0, "motorcycle": 0, "truck": 0, "pedestrian": 0, "other": 0}
        self.fps = 0
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.detection_model = None
        self.model_type = "yolo"  # Default model type
        self.next_object_id = 0
        self.speed_estimation_factor = 3.6  # Convert pixels/second to km/hour
        self.scale_factor = 8*20  # 1 square foot = 8x20 pixels
        
        # Create UI elements
        self.create_ui()
        
        # Load detection models
        self.load_models()
    
    def create_ui(self):
        # Create frames for organization
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        self.middle_frame = tk.Frame(self.root)
        self.middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        # Top frame - Buttons
        self.select_btn = Button(self.top_frame, text="Select Video", command=self.select_video)
        self.select_btn.pack(side=tk.LEFT, padx=10)
        
        self.start_btn = Button(self.top_frame, text="Start Detection", command=self.start_detection, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.reset_btn = Button(self.top_frame, text="Reset ROI", command=self.reset_roi)
        self.reset_btn.pack(side=tk.LEFT, padx=10)
        
        # Model selection
        self.model_label = Label(self.top_frame, text="Model:")
        self.model_label.pack(side=tk.LEFT, padx=10)
        
        self.model_var = tk.StringVar(value="yolo")
        self.yolo_radio = tk.Radiobutton(self.top_frame, text="YOLO", variable=self.model_var, value="yolo", command=self.change_model)
        self.yolo_radio.pack(side=tk.LEFT)
        
        self.rcnn_radio = tk.Radiobutton(self.top_frame, text="R-CNN", variable=self.model_var, value="rcnn", command=self.change_model)
        self.rcnn_radio.pack(side=tk.LEFT)
        
        # Middle frame - Video display
        self.canvas = tk.Canvas(self.middle_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Bottom frame - Status and counts
        self.status_label = Label(self.bottom_frame, text="Status: Idle")
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.count_label = Label(self.bottom_frame, text="Cars: 0 | Motorcycles: 0 | Trucks: 0 | Pedestrians: 0 | Other: 0")
        self.count_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = Label(self.bottom_frame, text="FPS: 0")
        self.fps_label.pack(side=tk.RIGHT, padx=10)
    
    def load_models(self):
        # Load YOLO model
        try:
            self.yolo_model = YOLO('yolov8x.pt')
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            
        # Load R-CNN model
        try:
            self.rcnn_model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
            self.rcnn_model.eval()
            if torch.cuda.is_available():
                self.rcnn_model = self.rcnn_model.cuda()
            print("R-CNN model loaded successfully")
        except Exception as e:
            print(f"Error loading R-CNN model: {e}")
        
        # Set default model
        self.detection_model = self.yolo_model
    
    def change_model(self):
        self.model_type = self.model_var.get()
        if self.model_type == "yolo":
            self.detection_model = self.yolo_model
        else:
            self.detection_model = self.rcnn_model
        print(f"Model changed to {self.model_type}")
    
    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
        if self.video_path:
            # Reset everything
            self.reset_roi()
            self.object_count = {"car": 0, "motorcycle": 0, "truck": 0, "pedestrian": 0, "other": 0}
            self.actual_object_count = {"car": 0, "motorcycle": 0, "truck": 0, "pedestrian": 0, "other": 0}
            self.tracked_objects = {}
            self.next_object_id = 0
            
            # Open video and display first frame
            self.cap = cv2.VideoCapture(self.video_path)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.display_frame(frame)
                self.status_label.config(text="Status: Video loaded. Select ROI by clicking on the frame.")
                self.start_btn.config(state=tk.NORMAL)
            else:
                self.status_label.config(text="Status: Error loading video.")
    
    def on_click(self, event):
        if self.detection_started or not self.current_frame is not None:
            return
            
        x, y = event.x, event.y
        
        # Adjust for potential scaling in display
        scale_x = self.frame_width / self.canvas.winfo_width()
        scale_y = self.frame_height / self.canvas.winfo_height()
        
        actual_x = int(x * scale_x)
        actual_y = int(y * scale_y)
        
        # Add point to ROI
        if len(self.roi_points) < 4:
            self.roi_points.append((actual_x, actual_y))
            # Draw point on canvas
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill="red")
            
            # Draw line if we have more than one point
            if len(self.roi_points) > 1:
                prev_x, prev_y = self.roi_points[-2]
                prev_x_scaled = prev_x / scale_x
                prev_y_scaled = prev_y / scale_y
                self.canvas.create_line(prev_x_scaled, prev_y_scaled, x, y, fill="red", width=2)
            
            # If we have 4 points, connect the last and first
            if len(self.roi_points) == 4:
                first_x, first_y = self.roi_points[0]
                first_x_scaled = first_x / scale_x
                first_y_scaled = first_y / scale_y
                self.canvas.create_line(x, y, first_x_scaled, first_y_scaled, fill="red", width=2)
                self.roi_selected = True
                self.status_label.config(text="Status: ROI selected. Click 'Start Detection' to begin.")
    
    def reset_roi(self):
        self.roi_points = []
        self.roi_selected = False
        if self.current_frame is not None:
            self.display_frame(self.current_frame)
        self.status_label.config(text="Status: ROI reset. Select new ROI.")
    
    def start_detection(self):
        if not self.roi_selected or not self.cap:
            self.status_label.config(text="Status: Please select ROI first.")
            return
            
        self.detection_started = True
        self.start_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Detection in progress...")
        
        # Process video in a separate thread to keep UI responsive
        import threading
        threading.Thread(target=self.process_video, daemon=True).start()
    
    def process_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning of video
        
        # Prepare video writer for saving results
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, self.fps, (self.frame_width, self.frame_height))
        
        # For speed calculation, we need to track objects across frames
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frame_count = 0
        
        # Performance metrics
        processing_times = []
        
        start_time = time.time()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Create copy for drawing
            display_frame = frame.copy()
            
            # Draw ROI
            cv2.polylines(display_frame, [np.array(self.roi_points)], True, (0, 255, 0), 2)
            
            # Create ROI mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            roi_np = np.array([self.roi_points], dtype=np.int32)
            cv2.fillPoly(mask, roi_np, 255)
            
            # Apply detection based on selected model
            frame_start_time = time.time()
            
            if self.model_type == "yolo":
                results = self.detect_yolo(frame, mask)
            else:
                results = self.detect_rcnn(frame, mask)
                
            # Calculate processing time for this frame
            frame_processing_time = time.time() - frame_start_time
            processing_times.append(frame_processing_time)
            
            # Update display frame with detections and counts
            display_frame = self.draw_detections(display_frame, results)
            
            # Update FPS calculation
            self.new_frame_time = time.time()
            fps = 1/(self.new_frame_time - self.prev_frame_time) if self.prev_frame_time > 0 else 0
            self.prev_frame_time = self.new_frame_time
            
            # Add FPS text to frame
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update progress
            processed_frame_count += 1
            progress = (processed_frame_count / total_frames) * 100
            
            # Update UI with current frame and progress
            self.current_frame = display_frame
            self.display_frame(display_frame)
            self.status_label.config(text=f"Status: Processing... {progress:.1f}%")
            self.count_label.config(text=f"Cars: {self.object_count['car']} | Motorcycles: {self.object_count['motorcycle']} | Trucks: {self.object_count['truck']} | Pedestrians: {self.object_count['pedestrian']} | Other: {self.object_count['other']}")
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            
            # Write frame to output video
            out.write(display_frame)
            
            # Update UI
            self.root.update()
        
        # Release resources
        out.release()
        
        # Calculate and display final results
        total_time = time.time() - start_time
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        final_status = f"Status: Completed in {total_time:.2f}s. Avg. frame processing time: {avg_processing_time*1000:.2f}ms"
        self.status_label.config(text=final_status)
        
        # Display final counts
        print("=== Detection Complete ===")
        print(f"Model: {self.model_type.upper()}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average frame processing time: {avg_processing_time*1000:.2f} ms")
        print("Final counts:")
        print(f"Cars: {self.object_count['car']}")
        print(f"Motorcycles: {self.object_count['motorcycle']}")
        print(f"Trucks: {self.object_count['truck']}")
        print(f"Pedestrians: {self.object_count['pedestrian']}")
        print(f"Other: {self.object_count['other']}")
        
        # Calculate accuracy if actual counts were provided
        if sum(self.actual_object_count.values()) > 0:
            accuracy = self.calculate_accuracy()
            print(f"Detection Accuracy: {accuracy*100:.2f}%")
        
        self.detection_started = False
        self.start_btn.config(state=tk.NORMAL)
    
    def detect_yolo(self, frame, mask):
        # Apply ROI mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Perform detection
        results = self.yolo_model(masked_frame)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                cls_name = self.yolo_model.names[cls]
                
                # Map YOLO classes to our categories
                category = self.map_to_category(cls_name)
                
                if conf > 0.5:  # Confidence threshold
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'category': category,
                        'confidence': conf
                    })
        
        return detections
    
    def detect_rcnn(self, frame, mask):
        # Apply ROI mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Convert to RGB for PyTorch
        rgb_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        x = torchvision.transforms.ToTensor()(rgb_frame)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            x = x.cuda()
            
        # Add batch dimension
        x = x.unsqueeze(0)
        
        # Perform detection
        with torch.no_grad():
            prediction = self.rcnn_model(x)
        
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
                category = self.map_to_category(label)
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'category': category,
                    'confidence': float(scores[i])
                })
        
        return detections
    
    def map_to_category(self, class_name):
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
    
    def draw_detections(self, frame, detections):
        # Track objects across frames for speed estimation and counting
        current_detections = {}
        
        for detection in detections:
            bbox = detection['bbox']
            category = detection['category']
            confidence = detection['confidence']
            
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Check if this detection matches any tracked object
            matched_id = None
            for obj_id, obj_data in self.tracked_objects.items():
                last_center = obj_data['centers'][-1]
                distance = math.sqrt((center_x - last_center[0])**2 + (center_y - last_center[1])**2)
                
                # If close enough, consider it the same object
                if distance < 50 and obj_data['category'] == category:
                    matched_id = obj_id
                    break
            
            # If no match, create new tracked object
            if matched_id is None:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.tracked_objects[obj_id] = {
                    'category': category,
                    'centers': [(center_x, center_y)],
                    'timestamps': [time.time()],
                    'counted': False
                }
                matched_id = obj_id
            else:
                # Update existing tracked object
                self.tracked_objects[matched_id]['centers'].append((center_x, center_y))
                self.tracked_objects[matched_id]['timestamps'].append(time.time())
            
            # Add to current detections
            current_detections[matched_id] = True
            
            # Calculate speed if enough positions are recorded
            speed = 0
            if len(self.tracked_objects[matched_id]['centers']) >= 2:
                # Calculate distance between last two points
                pts = self.tracked_objects[matched_id]['centers']
                times = self.tracked_objects[matched_id]['timestamps']
                
                # Use last 5 positions for better speed estimation, or fewer if not available
                num_points = min(5, len(pts))
                if num_points >= 2:
                    dx = pts[-1][0] - pts[-num_points][0]
                    dy = pts[-1][1] - pts[-num_points][1]
                    distance_pixels = math.sqrt(dx*dx + dy*dy)
                    
                    # Convert to real-world distance using scale factor
                    distance_meters = distance_pixels / self.scale_factor
                    
                    # Calculate time difference
                    time_diff = times[-1] - times[-num_points]
                    
                    if time_diff > 0:
                        # Speed in meters per second
                        speed_mps = distance_meters / time_diff
                        
                        # Convert to km/h
                        speed = speed_mps * self.speed_estimation_factor
            
            # Count object if it crosses the center of ROI and hasn't been counted yet
            if not self.tracked_objects[matched_id]['counted']:
                # Create a polygon from ROI points
                roi_poly = np.array(self.roi_points)
                
                # Check if the center is inside the ROI
                point_inside = cv2.pointPolygonTest(roi_poly, (center_x, center_y), False) >= 0
                
                if point_inside:
                    self.tracked_objects[matched_id]['counted'] = True
                    self.object_count[category] += 1
            
            # Determine color based on speed
            color = (0, 255, 0)  # Green by default
            if speed > 30:  # If speed exceeds 30 km/h
                color = (0, 0, 255)  # Red
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence and speed
            label = f"{category}: {confidence:.2f}, {speed:.1f} km/h"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Remove objects that are no longer detected
        objects_to_remove = []
        for obj_id in self.tracked_objects:
            if obj_id not in current_detections:
                # Check if object was inactive for too long
                if time.time() - self.tracked_objects[obj_id]['timestamps'][-1] > 1.0:  # 1 second threshold
                    objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
        
        # Draw object counts on the frame
        cv2.putText(frame, f"Car: {self.object_count['car']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Motors: {self.object_count['motorcycle']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Trucks: {self.object_count['truck']}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def display_frame(self, frame):
        # Resize the frame to fit the canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet, use default size
            canvas_width = 800
            canvas_height = 600
        
        # Calculate aspect ratio
        aspect_ratio = self.frame_width / self.frame_height
        
        # Calculate new dimensions
        if canvas_width / canvas_height > aspect_ratio:
            # Canvas is wider than frame
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            # Canvas is taller than frame
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Convert to RGB for tkinter
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.config(width=new_width, height=new_height)
        self.canvas.create_image(new_width/2, new_height/2, image=img_tk)
        self.canvas.image = img_tk  # Keep a reference
    
    def calculate_accuracy(self):
        total_actual = sum(self.actual_object_count.values())
        total_detected = sum(self.object_count.values())
        
        if total_actual == 0:
            return 1.0  # No actual objects, but we didn't detect any (perfect)
            
        # Calculate accuracy as 1 - (absolute difference / total actual)
        accuracy = 1.0 - (abs(total_detected - total_actual) / total_actual)
        return max(0.0, min(1.0, accuracy))  # Clamp between 0 and 1

def main():
    root = tk.Tk()
    app = VehicleDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()