from __future__ import absolute_import, annotations

import cv2 as cv
import numpy as np
from copy import deepcopy
from typing import List

from play_with_HSV import rescale_image, get_hsv_mask


image_path = "../media/20200402_104028_001.jpg"  # in the future we can set the path as argument or env var

ratio = 3  # we can try to set a high threshold instead of using this ratio
kernel_size = 3  # we can try to understand what is that

percentage = 85
scale = 1
delta = 0
depth = cv.CV_16S
n_derivative = 2


def grab_contours(contour_tuple: tuple) -> List[np.ndarray]:
    # in OpenCV v2.4, v4-official
    if len(contour_tuple) == 2:
        return contour_tuple[0]
    # in OpenCV v3
    elif len(contour_tuple) == 3:
        return contour_tuple[1]

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
    image_hsv = get_hsv_mask(image, cv.cvtColor(image, cv.COLOR_BGR2HSV), "g")
    greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blurred_image = cv.GaussianBlur(image, (3, 3), 0)
    greyscale_blurred_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)

    gradient_x = cv.Sobel(greyscale_blurred_image, depth, n_derivative, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    gradient_y = cv.Sobel(greyscale_blurred_image, depth, 0, n_derivative, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_gradient_x = cv.convertScaleAbs(gradient_x)
    abs_gradient_y = cv.convertScaleAbs(gradient_y)
    gradient_image = cv.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)
    cv.imshow("sobel gradient image", gradient_image)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(gradient_image, cv.MORPH_CLOSE, kernel)
    cv.imshow("closing after sobel", closing)
    open_after_closing = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    cv.imshow("open after closing", open_after_closing)

    positiveValuesArray = []
    for i, itemi in enumerate(open_after_closing):
        for j, itemj in enumerate(itemi):
            if itemj > 0:
                positiveValuesArray.append(itemj)
    percentile = np.percentile(positiveValuesArray, percentage)

    for i, itemi in enumerate(open_after_closing):
        for j, itemj in enumerate(itemi):
            if itemj <= percentile:
                open_after_closing[i, j] = 0
    cv.imshow("percentile removed", open_after_closing)
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
    greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edged = canny_threshold(greyscale_image, picked_threshold)
    bitwise_and = np.zeros((len(image), len(image[0])))
    for i, itemi in enumerate(edges_on_image):
        for j, itemj in enumerate(itemi):
            if itemj.any() > 0 and open_after_closing[i, j] > 0:
                bitwise_and[i, j] = int(255)
            else:
                bitwise_and[i, j] = int(0)
    bitwise_and = bitwise_and.astype(np.uint8)
    cv.imshow("bitwise", bitwise_and)
    bitwise_and_on_image = deepcopy(image_hsv)
    for i, item_i in enumerate(bitwise_and):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                bitwise_and_on_image[i][j] = (0, 0, 255)
    cv.imshow("bitwise_and_on_image", bitwise_and_on_image)

    contour_tuple = cv.findContours(bitwise_and.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contour_tuple)
    contour_image = bitwise_and.copy()
    area = 0

    for c in contours:
        area += cv.contourArea(c)
        cv.drawContours(contour_image, [c], 0, (100, 5, 10), 10)

    image_from_contours = deepcopy(image)
    for i, item_i in enumerate(contour_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                image_from_contours[i][j] = (0, 0, 0)
    cv.imshow("image_from_contours", image_from_contours)

    cv.imwrite("../test_images/bitwise_edges_on_image6.jpg", bitwise_and_on_image)
    cv.imwrite("../test_images/original_image6.jpg", image)
    cv.imwrite("../test_images/bitwise_canny_percentile6.jpg", bitwise_and)
    cv.imwrite("../test_images/hsv_image6.jpg", image_hsv)
    cv.imwrite("../test_images/bitwise_canny_percentile_area6.jpg", image_from_contours)

    cv.waitKey()
    cv.destroyAllWindows()
