# sprout-detection

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Project structure](#project-structure)
- [Setup](#setup)
  - [Installation](#installation)
  - [Required Python Packages](#required-python-packages)
- [Usage](#usage)
  - [color_picker](#color_picker)
  - [hsv_filter](#hsv_filter)
  - [main](#main)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

Project of signal, image and video @ University of Trento for the detection of sprouts.


## Project structure

    sprout-detection
    ├── (images)            [ignored folder where you can add your images and then change the "image.extension" in the files with the name of your images]
    └── src
        ├── image_utils               [package that contains the utils for images]
        |    └── standard_image_operations               [library which contains the main operations on images]
        ├── color_picker              [script to pick HSV colors and define ranges]
        ├── hsv_filter                [script to try our standard HSV filter on different kind of images]
        └── main                      [script to extract the relevant edges of the image using canny and sobel, also using an HSV filtering approach]


## Setup

### Installation

This repository can be cloned using the command:

```bash
    git clone https://github.com/samuelbortolin/sprout-detection.git
```


### Required Python Packages

Required Python packages can be installed using the command:

```bash
    pip install -r requirements.txt
```


## Usage

### color_picker

Change the path with the image to analyze and run the script:
* in the first phase it allows you to pick pixels from an image and then clicking `q` it returns the range of HSV values that contains all the selected pixels;
* in the second phase it allows you to build an HSV filter using trackbars to modify the value of hue, saturation and value, clicking a button it updates HSV filtered image and clicking `q` it returns the range of HSV values selected with the trackbars.

The script can be run using one of the two set of commands:

```bash
    cd src
    python3 color_picker.py
```

or

```bash
    cd src
    python3 -m color_picker
```

### hsv_filter

Change the path with the different types of images containing flowers, leaves or branches to analyze and run the script: it performs an isolation of the color for these images using our standard HSV filter.

The script can be run using one of the two set of commands:

```bash
    cd src
    python3 hsv_filter.py
```

or

```bash
    cd src
    python3 -m hsv_filter
```

### main

Change the path with the image to analyze and run the script: it extracts the relevant edges of the image using canny and sobel. It also tries an HSV color filtering approach and extracts the relevant edges of the HSV filtered image.

The script can be run using one of the two set of commands:

```bash
    cd src
    python3 main.py
```

or

```bash
    cd src
    python3 -m main
```
