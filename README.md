# Human-Pose-Estimation
This is an Undergraduate Final Year Project by Runze Cai. The goal of this FYP is to detect certain human behaviors based on OpenPose.

There are two main applications. The first one is Fall Detection, and the second is Bad Posture of Watching TV Detection.

You may start to run the two applications with the following guide, and you can e-mail me <cai@m.runze.xyz> if you meet any question.  

## Quick Start

### Get pre-trained OpenPose caffe model

Go to the folder `./model` and run `getModels.bat` or `getModels.sh` （depend on your OS).

### Get UR Fall Detection Dataset

Go to this [Website](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html) and download all the fall-`**`-cam0-rgb.zip files.

Put the zip files into the folder `./dataset/UR/` and unzip the images into `./dataset/UR/img`.

You may also download all the fall-`**`-cam0-rgb-d.zip files, and unzip the images into `./dataset/UR/depth`.

### Install Requirements

Make sure your environment has OpenCV, Keras, Numpy, Pandas, Sklearn, Matplotlib, etc.

You can install all the requirements by enter `pip install -r requirements.txt` in your terminal.

### Run the Program

Noted: be cautious about the `path variables` in files below.

#### Fall Detection

##### Step One

To begin with, please run `openpose_main.py` to get Human Skeleton Points from the video frames. You can check the results
in the `./output/fall_output_mytest`. You can get the results of rule-based fall detection as well.

At the same time, this python program generate `.npy` files which will be used later.

##### Step Two

Run `preprocess_points.py` to preprocess the Human Skeleton Points for later training.

##### Step Three

###### Skeleton Points Based Model

Run `train.py` to train the fall detection model based on skeleton points.

###### Depth Map Based Model

Run `depth_train.py` to train the fall detection model based on depth map.

###### Combination Model

Run `merge_train.py` to train the fall detection model with the combination of skeleton points and depth map.

##### Step Four

Run `fall_detection_demo.py` to run the demo of fall detection. 

The default model of fall detection in this program is model based on skeleton points. You may change the model if you like.

You may be required to input the information about e-mail of the sender and receiver when you run the program.
And the sender will e-mail the receiver when the program detects fall.

#### Bad Posture of Watching TV Detection

Run `watching_tv_pose_detection.py` to run the demo of bad posture of watching TV detection.

You may use the `video_to_img.py` program to convert your video to image frames first.



