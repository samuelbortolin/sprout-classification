# sprout-detection

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Setup](#setup)
  - [Required Python Packages](#required-python-packages)
  - [main.py](#mainpy)
  - [sobel_canny.py](#sobel_cannypy)
  - [color_picker.py](#color_pickerpy)
  - [hsv_filter.py](#hsv_filterpy)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

Project of signal, image and video @ Università degli studi di Trento for sprout classification


## Setup

### Required Python Packages

Required Python packages can be installed using the command:

```bash
pip install -r requirements.txt
```

### main.py

Change the path with the image to analyze and run the program to extract the relevant edges of the image. It also tries an hsv approach and extract edges on the hsv filtered image.

### sobel_canny.py

Change the path with the image to analyze and run the program to extract the relevant edges of the image using canny and sobel. It also tries an hsv approach.

### color_picker.py

Change the path with the image to analyze and run the program to in a first phase to select the pixels and get the values and in a second phase to build a filter.

### hsv_filter.py

Change the path with the images to analyze and run the program to filter via hsv different types of images.
