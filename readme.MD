![image](https://companieslogo.com/img/orig/NVDA_BIG-8c0fdc59.png?t=1633073585)

# Rear-Vehicle detection in real-time using HAAR Cascade Classifiers and Trackers

## Description

The "Rear-Vehicle Detection in Real-Time using HAAR Cascade Classifiers and Haar Cascade Classifiers and Trackers" project is aimed to track the back of car on Wyoming Highways using Jetson Nano SBC (Single Board Computer). This project uses various computer vision algorithms such as HAAR Cascade Classifiers, Object Tracking to detect the car from camera in real-time.

HAAR Cascade Classifiers are machine learning based object detection algorithm that detects an object in images and videos through training cascade function with positive and negative images. The positive images are the collection of images with the required features and negative images are the images without such features.

Such images are trained as *.xml* file and the trained file contains the dataset that is used for detecting objects in new images with similar features.

For this project, a custom Haar cascade classifier was developed using the collected dataset of lead car on various roads. The images where the lead car was visible were classified as positive images and stored in 'p' folder and the negative images, where the lead car was not visible were stored in \`n' folder. For developing a custom haar cascade classifier, **Cascade Trainer GUI** software was used.

A collection of 6,000 images were used for classification, where there were a total of almost 2,900 positive images and 3,100 negative images.

The trained *.xml* data were loaded as python code which are used to detect the car for the initial 5 seconds of the video. When a car is detected, the script takes the mean of the x co-ordinates, y co-ordinates, and height of the car. The height of the car remains similar to its value. Hence, the co-ordinates determined by the HAAR cascade, where the standard deviation of height is less, will be used as the co-ordinates for tracking. The script will initialize the tracking process, depending on what tracking method has been selected.

During the tracking part, the script keeps on updating the tracker with the information of incoming frames. If there is a significant change in the x y co-ordinates of the car, the script readjusts the tracking coordinates.

If the car goes completely out of frame, the tracking is stopped until the HAAR cascade detects a car again. Once a car is detected, the script takes the mean of the coordinates and initiates the tracking process once more.

## Requirements

Hardware Components:

* Jetson Nano
* CSI or USB Camera

Software Components:

* Terminal

## Running the Program

The program has been converted to an executable file using PyInstaller. Hence, the program can run independently without dependencies required.

Please enter the following commands in terminal for running the program on Jetson Nano:

```console
haar@jetson:~$ git clone nischalkhanal123/HAAR_Tracker.git
haar@jetson:~$ cd HAAR_Tracker
haar@jetson:~$ cd dist
haar@jetson:~$ cd HAAR_Tracker
haar@jetson:~$ ./HAAR_Tracker
```

After running the executable file, the following window will show up:

![image](https://imgtr.ee/images/2023/06/02/SuHqB.png)

## GUI Functions

###### Stream Camera

This button runs the camera. If it gives error, make sure that camera is connected properly

###### Select Tracker

This drop-down menu lets you select the tracker. The default tracker is **MOSSE** as it has been tested to work better than the other trackers, but depending on the surrounding and camera, it may change.

Hence, other trackers can be selected and tested to see if they work better.

###### Button 'q'

As given in the GUI description, if you are not satisfied with the current tracker, you can press q to close the program, select a new tracker and start stream.

###### Exit

This button safely exits the overall GUI, after shutting the camera.