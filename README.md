# Football-Analysis-System
# Football Analysis System

## Overview

An end-to-end Computer Vision pipeline for football match analytics built using YOLOv11x, ByteTrack, OpenCV, and TensorFlow.

The system analyzes match footage by detecting and tracking players, referees, and the ball while generating performance metrics such as speed estimation, distance traveled, and team classification.

---

## Features

* Player Detection
* Ball Detection
* Referee Detection
* Multi-Object Tracking using ByteTrack
* Speed Estimation
* Distance Traveled Estimation
* Camera Motion Compensation
* Team Classification using K-Means Clustering

---

## Workflow

Video Input
→ Object Detection (YOLOv11x)
→ Object Tracking (ByteTrack)
→ Camera Motion Estimation
→ Speed & Distance Analysis
→ Team Classification
→ Annotated Video Output

---

## Tech Stack

* Python
* OpenCV
* Ultralytics YOLOv11x
* TensorFlow
* NumPy

---

## Key Learnings

* Object Detection
* Multi-Object Tracking
* Sports Analytics
* Computer Vision Pipelines
* Clustering Algorithms
* Motion Analysis

---

## Results

Successfully tracked players, referees, and football across video frames while generating analytical insights including speed estimation and team classification.

---

## Future Improvements

* Expected Goals (xG) Analysis
* Pass Network Visualization
* Heatmaps
* Real-Time Inference
* Tactical Formation Detection

---

## Repository Structure

```text
project/
│
├── detection.py
├── tracking.py
├── speed_estimator.py
├── camera_motion.py
├── team_classifier.py
├── main.py
└── README.md
```
