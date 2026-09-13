# Traffic Monitoring and Vehicle Tracking

A computer vision project for monitoring traffic conditions in video footage using YOLO, OpenCV, and object tracking.

The system detects and tracks vehicles in a video, counts vehicles crossing a predefined line, estimates traffic occupancy and movement speed, and classifies the overall traffic condition into three categories:

- **Normal Traffic**
- **Heavy Traffic**
- **Traffic Jam**

## Features

- Vehicle detection using YOLOv8
- Vehicle tracking using ByteTrack
- Detection of cars, motorcycles, buses, and trucks
- Vehicle counting using a virtual counting line
- Approximate movement-speed estimation
- Traffic occupancy analysis
- Smoothed traffic-condition classification
- Real-time visualization using OpenCV

## How It Works

The program processes the selected video frame by frame.

For each frame, YOLO detects the vehicles and ByteTrack assigns tracking IDs to them. The program keeps a short history of each vehicle's position to estimate its movement.

A virtual horizontal line is used to count vehicles when they cross it. The system also estimates how much of the frame is occupied by vehicles and combines this information with their estimated movement speed to calculate a traffic index.

Based on the smoothed traffic index, the system classifies the traffic condition as **Normal Traffic**, **Heavy Traffic**, or **Traffic Jam**.

## Sample Videos

The repository includes four sample videos for testing the system. Each sample video can demonstrate one of the three supported traffic conditions:

- **Normal Traffic**: The number and density of vehicles are relatively low, and vehicles are moving at a normal speed.
- **Heavy Traffic**: The number and density of vehicles are higher, but vehicles are still moving.
- **Traffic Jam**: Vehicle density is high, and vehicles are moving very slowly or are almost stationary

These videos are provided as examples, but the program is not limited to them. Other traffic videos can also be tested with the system, as long as they are compatible with the expected video input.

## Presentation

A Persian PowerPoint presentation explaining the project is available in the `Presentation` folder.

## Requirements

- Python 3.x
- OpenCV
- Ultralytics YOLO

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Usage

Place the videos in the `videos` directory and make sure their names match the entries in the `video_map` dictionary.

Run the program with:

```bash
python traffic_monitor.py
```

Select a video when prompted:

```text
Select video to process:
1 -> video 1
2 -> video 2
3 -> video 3
4 -> video 4
```

The processed video will be displayed in a window with the detected vehicles, vehicle count, traffic status, and counting line.

Press **Esc** to stop the program.

## Project Structure

```text
traffic-monitoring-yolo/
│
├── traffic_monitor.py
├── README.md
├── requirements.txt
├── Presentation/
│   └── project_presentation.pdf
├── videos/
│   ├── 1.mp4
│   ├── 2.mp4
│   ├── 3.mp4
│   └── 4.mp4
├── output/
│   └── output.mp4
└── .gitignore
```

## Technologies

- Python
- OpenCV
- YOLOv8
- ByteTrack
- Computer Vision
- Object Detection
- Object Tracking

## Notes

The estimated speed and traffic index are based on visual information from the video rather than real-world calibrated measurements. Therefore, the traffic classification is intended for computer vision analysis and demonstration rather than precise traffic engineering measurements.
