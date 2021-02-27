from __future__ import absolute_import, annotations

import cv2 as cv
import numpy as np


image_flowers_path = "../media/ROBI_VT01_20200507_3.png"
image_leaves_path = "../media/FOTO1ROBI_MI02_20200422.jpg"
image_branches_path = "../media/image.jpg"


def apply_mask(original_image: np.ndarray, hsv_image: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    # get the image after applying hsv mask based on a range of hsv colors

    mask = cv.inRange(hsv_image, lower_bound, upper_bound)
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, (5, 5), iterations=1)
    return cv.bitwise_and(original_image, original_image, mask=mask)


def get_hsv_mask(original_image: np.ndarray, hsv_image: np.ndarray, color: str) -> np.ndarray:
    # get the image after applying hsv mask based on color of interest

    if color == "w":
        lower_bound = np.array([15, 0, 100])
        upper_bound = np.array([35, 40, 255])
    elif color == "g":  # values to be estimated using the color_picker and then tested with play_with_HSV
        lower_bound = np.array([20, 0, 0])
        upper_bound = np.array([80, 255, 255])
    elif color == "b":  # values to be estimated using the color_picker and then tested with play_with_HSV
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([30, 255, 255])
    else:
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 255])
    return apply_mask(original_image, hsv_image, lower_bound, upper_bound)


def rescale_image(image_to_rescale: np.ndarray, target_number_of_pixels: int = 100000) -> np.ndarray:
    # rescale image all to the same standard number of pixels

    scale = (len(image_to_rescale[0]) * len(image_to_rescale) / target_number_of_pixels) ** (1 / 2)
    return cv.resize(image_to_rescale, (0, 0), fx=(1 / scale), fy=(1 / scale))


if __name__ == "__main__":

    # x flowers
    image = cv.imread(image_flowers_path)
    image = rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("result after applying hsv mask", get_hsv_mask(image, hsv_image, "w"))
    cv.waitKey()
    cv.destroyAllWindows()

    # x leaves
    image = cv.imread(image_leaves_path)
    image = rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("result after applying hsv mask", get_hsv_mask(image, hsv_image, "g"))
    cv.waitKey()
    cv.destroyAllWindows()

    # x branches
    image = cv.imread(image_branches_path)
    image = rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("result after applying hsv mask", get_hsv_mask(image, hsv_image, "b"))
    cv.waitKey()
    cv.destroyAllWindows()
