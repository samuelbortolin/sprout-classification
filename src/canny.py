from __future__ import absolute_import, annotations

import cv2 as cv
import numpy as np

from play_with_HSV import rescale_image, get_hsv_mask


image_path = "../media/IMG_20200324 LC01a BASSANI.jpg"  # in the future we can set the path as argument or env var

ratio = 3  # we can try to set a high threshold instead of using this ratio
kernel_size = 3  # we can try to understand what is that

scale = 1
delta = 0
depth = cv.CV_16S
n_derivative = 2


def canny_threshold(greyscale_image: np.ndarray, low_threshold: int) -> np.ndarray:
    # apply canny and return the edges

    blurred_image = cv.blur(greyscale_image, (3, 3))
    return cv.Canny(blurred_image, low_threshold, low_threshold * ratio, kernel_size)


def canny_threshold_mask(greyscale_image: np.ndarray, low_threshold: int) -> int:
    # apply canny and return the sum of elements in the mask

    blurred_image = cv.blur(greyscale_image, (3, 3))
    detected_edges = cv.Canny(blurred_image, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    return np.ma.sum(mask)


def canny_threshold_with_image(greyscale_image: np.ndarray, low_threshold: int, original_image: np.ndarray) -> np.ndarray:
    # apply canny and return the edges on the image

    blurred_image = cv.blur(greyscale_image, (3, 3))
    detected_edges = cv.Canny(blurred_image, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    return original_image * (mask[:, :, None].astype(original_image.dtype))


if __name__ == "__main__":

    image = cv.imread(image_path)
    image = rescale_image(image)
    greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    color = input("What do you want to analyze? Write w for white/flowers, g for green/leaves or b for brown/branches: ")
    hsv_filtered_image = get_hsv_mask(image, cv.cvtColor(image, cv.COLOR_BGR2HSV), color)
    blurred_image = cv.GaussianBlur(hsv_filtered_image, (3, 3), 0)
    greyscale_blurred_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)

    gradient_x = cv.Sobel(greyscale_blurred_image, depth, n_derivative, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    gradient_y = cv.Sobel(greyscale_blurred_image, depth, 0, n_derivative, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_gradient_x = cv.convertScaleAbs(gradient_x)
    abs_gradient_y = cv.convertScaleAbs(gradient_y)
    cv.imshow("abs_grad_x", abs_gradient_x)
    cv.imshow("abs_grad_y", abs_gradient_y)
    gradient_image = cv.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)
    cv.imshow("sobel gradient image", gradient_image)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(gradient_image, cv.MORPH_CLOSE, kernel)
    cv.imshow("closing after sobel", closing)
    open_after_closing = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    cv.imshow("open after closing", open_after_closing)
    open_after_sobel = cv.morphologyEx(gradient_image, cv.MORPH_OPEN, kernel)
    cv.imshow("open after sobel", open_after_sobel)
    close_after_opening = cv.morphologyEx(open_after_sobel, cv.MORPH_CLOSE, kernel)
    cv.imshow("close after opening", close_after_opening)
    # since the edges aren't always really clear it is more effective to do closing first and then opening, otherwise we end up with disconnected edges

    # try a edge detector approach
    previous_threshold = canny_threshold_mask(greyscale_image, 25)
    previous_delta = canny_threshold_mask(greyscale_image, 24) - previous_threshold
    picked_threshold = 0
    for i in range(26, 1000):
        new_threshold = canny_threshold_mask(greyscale_image, i)
        delta = previous_threshold - new_threshold
        previous_threshold = new_threshold
        if (delta + previous_delta) < 50:
            picked_threshold = i
            break
        previous_delta = delta

    # show the original and the resulting image
    cv.imshow("original image", image)
    edges_on_image = canny_threshold_with_image(greyscale_image, picked_threshold, image)
    cv.imshow("relevant edges of the image", edges_on_image)
    cv.waitKey()
    cv.destroyAllWindows()
