# Human-Pose-Estimation
This is a Undergraduate Final Year Project at UCAS. The goal of this FYP is to detect certain human behaviors based on OpenPose.

## Quick Start

### Get pre-trained OpenPose caffe model

Go to the folder `./model` and run `getModels.bat` or `getModels.sh` （depend on your OS).

### Get UR Fall Detection Dataset

Go to this [website](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html) and download all the fall-`**`-cam0-rgb.zip files. 

Put the zip files into the folder `./dataset/UR/` and unzip the images into `./dataset/UR/img`.

### Install Requirements

Make sure your environment has OpenCV, Keras, Numpy, Pandas, Sklearn, Matplotlib, etc.

You can install all the requirements by enter `pip install -r requirements.txt` in your terminal.

### Run the Program

#### Step One

To begin with, please run `openpose_main.py` to get Human Skeleton Points from the video frames. You can check the results
in the `./output/fall_output_mytest`. You can get the results of rule-based fall detection as well.

At the same time, this python program generate `.npy` files which will be used later.

#### Step Two

Run `preprocess_points.py` to preprocess the Human Skeleton Points for later training.

#### Step Three

Run `train.py` to train the fall detection model.


