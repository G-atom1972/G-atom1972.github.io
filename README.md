# **Gesture Based Navigation System**

## **Project Overview**

    Modern digital environments require non-contact interaction, especially in contexts such as dynamic presentations, sterile medical settings, or public touchless kiosks. 
    Traditional input devices like mice and keyboards are often limiting, unhygienic in shared spaces, and inefficient where spatial freedom is required.

This project implements a high-precision, real-time gesture recognition system designed to enable seamless, touchless navigation in specialized digital environments.

## **Key Features**

    Touchless Interaction: Navigate digital interfaces without physical contact.

    Real-time Recognition: Low-latency processing for immediate feedback.

    High Accuracy: Achieves over 94% accuracy in gesture detection.

    Lightweight Architecture: Designed to run efficiently using standard camera hardware.

## **Tech Stack**

    **Model:** YOLOv11 (Custom trained for specific gestures)

    **Computer Vision:** OpenCV, MediaPipe

    **Language:** Python

    **Dataset Management:** RoboFlow

    **Dashboard UI:** HTML5, Tailwind CSS, JavaScript (Canvas API)

## **Methodology**

### **1. Dataset / Input**

    The system utilizes a custom dataset obtained from RoboFlow, which was manually annotated to ensure high precision for specific navigation gestures. 
    Preprocessing involves resizing, normalization, and augmentation to ensure robustness in varying lighting conditions.

### **2. Model Architecture**

    The core logic resides in a custom YOLOv11 machine learning model. 
    This model works in tandem with MediaPipe for skeletal tracking and OpenCV for image processing, 
    allowing the system to map physical hand movements to digital navigation commands (e.g., scroll, click, swipe).

### **3. Performance**

    Accuracy: 94.2%

    Inference Speed: Optimized for real-time execution on standard CPUs/GPUs.

### **Literature Review**

    O. M. Vultur, Ş. G. Pentiuc and A. Ciupu, "Navigation system in a virtual environment by gestures," 
    2012 9th International Conference on Communications (COMM), Bucharest, Romania, 2012, pp. 111-114.

This research highlights the transition from high-latency sensor-based systems (like the original Kinect) to lightweight, camera-driven architectures. This project bridges the gap identified in early research by utilizing modern object detection models (YOLOv11) for superior speed and accuracy.

## **Academic Credits**

    Institution: Manipal University Jaipur

    Department: Computer Science & Engineering

    Year: 2026 (PBL - Project Based Learning)

    Student: Gautam Sahu (Reg No: 2427030540)

    Project Guide: Dr. Aditya Sinha (Assistant Professor)
