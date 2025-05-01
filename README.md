# Vehicle Detection, Classification and Speed Estimation System

![Vehicle Detection System UI](screenshots/main_ui.png)

A modern computer vision system that detects, classifies, tracks, and estimates the speed of vehicles in traffic videos. Built with YOLOv8 and Faster R-CNN, this project provides a comprehensive solution for traffic monitoring and analysis.

## Features

- **Multi-Model Detection**: Implements both YOLOv8 and Faster R-CNN for comparison
- **Vehicle Classification**: Categorizes vehicles as Cars, Motorcycles, Trucks, Pedestrians, and Others
- **No-Repetition Tracking**: Ensures each vehicle is counted only once
- **Speed Estimation**: Estimates vehicle speed with color-coded warnings when speeds exceed 30 km/h
- **Interactive ROI Selection**: Define custom regions of interest with a user-friendly interface
- **Real-Time Statistics**: View live counts, processing metrics, and performance data
- **Modern UI**: Beautiful, responsive interface with animations and visual feedback
- **Docker Deployment**: Easy deployment via containerization

## Demo

![Vehicle Detection Demo](screenshots/demo.gif)

## System Architecture

The system follows a modular architecture:

1. **Video Input Module**: Handles video loading and frame extraction
2. **Object Detection Module**: Implements detection models
3. **Object Tracking Module**: Maintains vehicle identity across frames
4. **Speed Estimation Module**: Calculates vehicle speeds
5. **User Interface Module**: Provides web-based interface
6. **Deployment Module**: Enables containerization

## Performance Comparison

| Metric | YOLOv8n | Faster R-CNN |
|--------|---------|--------------|
| Processing Speed (FPS) | 8.5 | 2.3 |
| Processing Time (s) for 1-min video | 98.6 | 267.4 |
| Detection Accuracy | 89.2% | 93.5% |
| Memory Usage (MB) | 450 | 780 |
| GPU Utilization | 65% | 92% |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional but recommended)

### Option 1: Docker Installation (Recommended)

1. Install [Docker](https://docs.docker.com/get-docker/)
2. Clone this repository:
3. git clone https://github.com/fahad-jameel/vhicle-Detection-system-using-yolo-and-FasterRCNN.git
cd vehicle-detection-system
3. Build the Docker image:
docker build -t vehicle-detection-system .
4. Run the container:
docker run -p 5000:5000 -v $(pwd)/uploads:/app/uploads vehicle-detection-system
On Windows PowerShell, use:
docker run -p 5000:5000 -v ${PWD}/uploads:/app/uploads vehicle-detection-system

### Option 2: Manual Installation

1. Clone this repository:
git clone https://github.com/fahad-jameel/vhicle-Detection-system-using-yolo-and-FasterRCNN.git
cd vehicle-detection-system
2. Create a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies:
pip install -r requirements.txt
4. Run the application:
python app.py

## Usage

1. Open your browser and navigate to `http://localhost:5000`
2. Click "Select Video" to upload a traffic video
3. Define a Region of Interest (ROI) by clicking four points on the video
4. Choose a detection model (YOLO or R-CNN)
5. Click "Start Detection" to begin processing
6. View real-time statistics and download processed video when complete

![ROI Selection](screenshots/roi_selection.png)



## Optimization Techniques

The system implements several optimization techniques:

- **Frame Resizing**: Reduces dimensions for faster processing
- **Batch Processing**: Processes multiple frames at once
- **ROI Masking**: Only processes the defined region of interest
- **Model Selection**: Uses smaller, faster models when appropriate
- **GPU Acceleration**: Leverages CUDA when available

## Results and Evaluation

The system was evaluated using various traffic videos with known vehicle counts:

- **Counting Accuracy**: 90.4% (YOLO) and 96.4% (R-CNN)
- **Speed Estimation Error**: Average of 6% compared to ground truth
- **Processing Performance**: Up to 8.5 FPS with optimizations enabled

![Results Dashboard](screenshots/results_dashboard.png)

## Future Improvements

- Multi-camera support
- Advanced tracking algorithms (SORT/DeepSORT)
- Automated ROI selection
- 3D speed estimation with calibration
- Cloud deployment
- Traffic pattern analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLOv8 implementation by [Ultralytics](https://github.com/ultralytics/ultralytics)
- Faster R-CNN implementation from [PyTorch](https://pytorch.org/)
