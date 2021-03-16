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
        |    └── image_operations               [library which contains the operations on images]
        ├── color_picker              [script to pick HSV colors and define ranges]
        ├── hsv_filter                [script to filter different kind of images]
        └── main                      [script to extract the relevant edges of the image using canny and sobel, also using a hsv filtering approach]


## Setup

### Installation

This repository can be clones using the command:

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

Change the path with the image to analyze and run the script to:
* in a first phase to select the pixels and get the values;
* in a second phase to build a filter.

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

Change the path with the different types of images to analyze and run the program to filter via hsv these images.

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

Change the path with the image to analyze and run the program to extract the relevant edges of the image using canny and sobel. It also tries a hsv filtering approach and extracts edges on the hsv filtered image.

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
