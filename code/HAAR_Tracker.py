'''
        Real-time Rear-Vehicle detection using HAAR Cascade Classifiers and OpenCV Trackers in combination.
        Libraries Required: OpenCV-contrib (this automatically installs numpy), PySimpleGUI
        Edge Devices Supported: Devices that runs Linux Distributions
        Tested on: Raspberry Pi 4B, NVIDIA Jetson Nano
        *******************************************************
        Author: Nischal Khanal
        Team: HPC Research Group
        Department: Electrical Engineering and Computer Science
        Institute: University of Wyoming
        ********************************************************
'''
import cv2
import PySimpleGUI as sg
import numpy as np
import threading

# Load the HAAR cascade XML file for car detection
car_cascade = cv2.CascadeClassifier('C://Users//nkhanal//Desktop//cars.xml')

#Initialize the type of tracker
def initialize_tracker(frame, bbox, tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    return tracker
# Detect the car in a frame
def detect_car(frame, car_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))
    return cars

# Get the mean values of the detected cars in frame
def get_mean_values(cars):
    x_coords = [car[0] for car in cars]
    y_coords = [car[1] for car in cars]
    heights = [car[3] for car in cars]
    mean_x = int(np.mean(x_coords))
    mean_y = int(np.mean(y_coords))
    mean_height = int(np.mean(heights))
    return mean_x, mean_y, mean_height

# Track the car in frame
def track_car(frame, tracker):
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = [int(coord) for coord in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return frame, bbox, success

# Sub-main function to run the vehicle detection and tracking
def main2(tracker_type):
    car_cascade = cv2.CascadeClassifier('cars.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    tracker = None
    car_detected = False
    bbox = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if not car_detected:
            cars = detect_car(frame, car_cascade)
            if len(cars) > 0:
                bbox = tuple(cars[0])
                tracker = initialize_tracker(frame, bbox, tracker_type)
                car_detected = True
        else:
            frame, new_bbox, success = track_car(frame, tracker)
            if not success:
                tracker = None
                car_detected = False
        cv2.imshow('Car Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Main function that runs the GUI
def main():
    # Define the GUI layout
    layout = [
        [sg.Text('Real-time Vehicle Detection', justification='center', size=(30, 1), font=('Arial', 15))],
        [sg.Button('Stream Camera', size=(15, 2), pad=((100, 50), (20, 20)))],
        [sg.Text('Select Tracker:'), sg.Combo(['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT'], default_value='CSRT', key='tracker_selector')],
        [sg.Text("To select a new tracker for the video, ",font=('Tahoma', 12))],
        [sg.Text("press 'q' to exit the camera window", font=('Tahoma', 12))],
        [sg.Button('Exit', size=(15, 2), pad=((100, 50), (20, 20)))],
    ]
    window = sg.Window('Computer Vision Operations', layout)
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        if event == 'Stream Camera':
            video_path = cv2.VideoCapture(0)
            ret, frame = video_path.read()
            if ret:
                tracker_type = values['tracker_selector']
                main2(tracker_type)
    window.close()
if __name__ == '__main__':
    main()