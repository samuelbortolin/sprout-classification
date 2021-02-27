from __future__ import absolute_import, annotations

from copy import deepcopy
from typing import List

import cv2 as cv
import numpy as np
"""
since the edges aren't always really clear it is more effective to do closing first and then opening, otherwise we end up with disconnected edges
"""
def rescale_image(image_to_rescale: np.ndarray, target_number_of_pixels: int = 100000) -> np.ndarray:
    scale = (len(image_to_rescale[0]) * len(image_to_rescale) / target_number_of_pixels) ** (1 / 2)
    # rescale image all to the same standard number of pixels
    return cv.resize(image_to_rescale, (0, 0), fx=(1 / scale), fy=(1 / scale))

def canny_threshold_mask(low_threshold: int) -> int:
    blurred_image = cv.blur(greyscale_image, (3, 3))
    detected_edges = cv.Canny(blurred_image, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    return np.ma.sum(mask)


def canny_threshold_with_image(low_threshold: int) -> np.ndarray:
    blurred_image = cv.blur(greyscale_image, (3, 3))
    detected_edges = cv.Canny(blurred_image, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    return image * (mask[:, :, None].astype(image.dtype))

ratio = 3  # we can try to set a high threshold instead of using this ratio
kernel_size = 3  # we can try to understand what is that

scale = 1
delta = 0
ddepth = cv.CV_16S

image = cv.imread("../media/IMG_20200324 LC01a BASSANI.jpg")  # in the future we can set the path as argument or env var
image = rescale_image(image)
image_blurred = cv.GaussianBlur(image, (3, 3), 0)
greyscale_image_blurred = cv.cvtColor(image_blurred, cv.COLOR_BGR2GRAY)
greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
nderivate = 2

if __name__ == "__main__":
    grad_x = cv.Sobel(greyscale_image_blurred, ddepth, nderivate, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(greyscale_image_blurred, ddepth, 0, nderivate, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    cv.imshow("abs_grad_x", abs_grad_x)
    cv.imshow("abs_grad_y", abs_grad_y)
    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv.imshow("sobel", grad)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(grad, cv.MORPH_CLOSE, kernel)
    cv.imshow("closing_after_sobel", closing)
    open_after_closing = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    cv.imshow("open_after_closing", open_after_closing)
    open_after_sobel = cv.morphologyEx(grad, cv.MORPH_OPEN, kernel)
    cv.imshow("open_after_sobel", open_after_sobel)
    close_after_opening = cv.morphologyEx(open_after_sobel, cv.MORPH_CLOSE, kernel)
    cv.imshow("close_after_opening", close_after_opening)

    # try a edge detector approach
    previous_threshold = canny_threshold_mask(25)
    previous_delta = canny_threshold_mask(24) - previous_threshold
    picked_threshold = 0
    for i in range(26, 1000):
        new_threshold = canny_threshold_mask(i)
        delta = previous_threshold - new_threshold
        previous_threshold = new_threshold
        if (delta + previous_delta) < 50:
            picked_threshold = i
            break
        previous_delta = delta

    # show the original and the resulting image
    cv.imshow("original image", image)
    edges_on_image = canny_threshold_with_image(picked_threshold)
    cv.imshow("relevant edges of the image", edges_on_image)
    cv.waitKey()