# Grape Ripeness Detection using YOLO and OpenCV

This project implements a grape ripeness detection system using a YOLO model with OpenCV for real-time object detection. It also includes object tracking, serial communication, and a user interface to display detections.

## Features
- Real-time grape ripeness detection using a YOLO model.
- Object tracking to associate detected objects across frames.
- Serial communication to send detection results.
- Adjustable detection mode (full-frame or center-focused detection).
- FPS display for performance monitoring.

## Requirements
### Hardware
- Webcam (for real-time video capture)
- Computer with a GPU (optional for faster YOLO inference)
- Serial device (e.g., Arduino) for receiving detection results (optional)

### Software
- Python 3.8+
- OpenCV
- Ultralytics YOLO
- NumPy
- Torch
- Serial (pyserial)

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your_username/grape-ripeness-detection.git
   cd grape-ripeness-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install required libraries manually (if needed):
   ```bash
   pip install opencv-python numpy torch ultralytics pyserial
   ```
4. Download the YOLO model from the following link and place it in the specified path:
   [Download Model](https://drive.google.com/file/d/1rQNjqBI9jcxcGTvf17JREvY3DTCxbavu/view?usp=drive_link)

5. Run the script:
   ```bash
   python main.py --webcam-resolution 640 480 --confidence 0.5 --model-path path/to/your/model.pt
   ```

## Usage
### Key Controls
- `Esc` - Exit the program
- `Space` - Toggle detection mode (full-frame vs. center detection)

## Object Tracking
The object tracker maintains detected objects for a configurable number of frames before they are considered lost. It uses an Intersection over Union (IoU) algorithm to associate new detections with existing objects.

## Serial Communication
- The program attempts to open a serial connection to a predefined COM port (`COM8` by default).
- If ripe grapes are detected, it sends `Yes` over the serial port; otherwise, it sends `No`.
- The system includes an automatic serial port reconnection feature.

## Troubleshooting
### Camera Not Opening
- Ensure the webcam is properly connected and not in use by another application.
- Try using a different camera index in `cv2.VideoCapture(index)`.

### Model Not Loading
- Check that the model file exists at the specified path.
- Ensure the correct YOLO model format is used.

### Serial Port Issues
- Verify that the correct COM port is specified.
- Ensure the serial device is connected and recognized by the system.

## Contributing
Feel free to fork this repository and submit pull requests with improvements or bug fixes.

## License
This project is licensed under the MIT License.

## Future Enhancements
- Implement multi-threading for faster inference.
- Add a graphical user interface (GUI) for better user experience.
- Enhance the tracking algorithm for improved accuracy.
- Expand detection to include different fruit types.

[Download YOLO Model](https://drive.google.com/file/d/1rQNjqBI9jcxcGTvf17JREvY3DTCxbavu/view?usp=drive_link)

