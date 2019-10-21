# Computer Vision Basics


## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Result](#result)
* [Sources](#sources)


## General info
This project was an assignment to predict images classification based on a training set labeled into 15 groups. Classification was done by using KNN and SVM to predict images based on their SIFT, SURF, and ORB features. Additionally, a straight pixel comparison on shrunken images was performed as a baseline for accuracy.


## Setup
* Have a version of Python3 installed (I used 3.6.8)
* Install necessary packages to run the program

```
$ pip install opencv-contrib-python==3.4.2.16
$ pip install -U scikit-learn
```

* Run the program
```
$ cd code/
$ python homework1.py
```


## Result
* Program should print out accuracies and times for completion to the terminal (do not be surprised if this takes multiple hours to complete)
* Program should store output values into .npy files under ./code/Results


## Sources
* https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/ 
* https://stackoverflow.com/questions/40645985/opencv-python-normalize-image/42164670
* https://www.datacamp.com/community/tutorials/
* k-nearest-neighbor-classification-scikit-learn
* https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
* https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html
* https://stackoverflow.com/questions/8220801/how-to-use-timeit-module
* https://pysource.com/2018/03/21/
* feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/