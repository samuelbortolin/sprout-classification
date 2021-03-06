from __future__ import absolute_import, annotations

from copy import deepcopy
from typing import List

import cv2 as cv
import numpy as np

from play_with_HSV import apply_mask, rescale_image


ratio = 3  # we can try to set a high threshold instead of using this ratio
kernel_size = 3  # we can try to understand what is that

image = cv.imread("../media/20200402_104001.jpg")  # in the future we can set the path as argument or env var
image = rescale_image(image)
greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#color = input("What do you want to analyze? Write w for white/flowers, g for green/leaves or b for brown/branches\n")


def get_hsv_mask(original_image: np.ndarray, frame_hsv: np.ndarray) -> None:
    if color == "w":
        lower_bound = np.array([15, 40, 100])
        upper_bound = np.array([35, 255, 255])
    elif color == "g":  # values to be estimated using the color_picker and then tested with play_with_HSV
        lower_bound = np.array([35, 100, 100])
        upper_bound = np.array([80, 255, 255])
    elif color == "b":  # values to be estimated using the color_picker and then tested with play_with_HSV
        lower_bound = np.array([30, 0, 0])
        upper_bound = np.array([50, 75, 200])
    else:
        lower_bound = np.array([0, 0, 0])
        upper_bound = np.array([180, 255, 255])
    apply_mask(deepcopy(original_image), frame_hsv, lower_bound, upper_bound)


def grab_contours(contour_tuple: tuple) -> List[np.ndarray]:
    # in OpenCV v2.4, v4-official
    if len(contour_tuple) == 2:
        return contour_tuple[0]
    # in OpenCV v3
    elif len(contour_tuple) == 3:
        return contour_tuple[1]


def canny_threshold(low_threshold: int) -> np.ndarray:
    blurred_image = cv.blur(greyscale_image, (3, 3))
    return cv.Canny(blurred_image, low_threshold, low_threshold * ratio, kernel_size)


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


if __name__ == "__main__":
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

    # try to connect open edges with contours
    edged = canny_threshold(picked_threshold)
    contour_tuple = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contour_tuple)
    contour_image = edged.copy()
    area = 0

    for c in contours:
        area += cv.contourArea(c)
        cv.drawContours(contour_image, [c], 0, (100, 5, 10), 10)

    image_from_contours = deepcopy(image)
    for i, item_i in enumerate(contour_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                image_from_contours[i][j] = (0, 0, 0)
    cv.imshow("area", image_from_contours)

    # show original image with the edges marked with a red line
    image_and_edges = deepcopy(image)
    for i, item_i in enumerate(edges_on_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                image_and_edges[i][j] = (0, 0, 255)
    cv.imshow("edges on original image", image_and_edges)

    # try color after edges
    get_hsv_mask(image_and_edges, cv.cvtColor(image_and_edges, cv.COLOR_BGR2HSV))

    # try a segmentation approach
    # ret, thresh = cv.threshold(greyscale_image, 0, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2)
    thresh = cv.adaptiveThreshold(greyscale_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    opening = cv.morphologyEx(closing, cv.MORPH_OPEN, kernel)
    image_with_threshold_area = deepcopy(image)
    for i, item_i in enumerate(opening):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                image_with_threshold_area[i][j] = (0, 0, 0)
    cv.imshow("image after threshold closed image", image_with_threshold_area)

    # show threshold image with the edges marked with a red line
    bgr_thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    for i, item_i in enumerate(edges_on_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                bgr_thresh[i][j] = (0, 0, 255)
    cv.imshow("edges on threshold image", bgr_thresh)

    # noise removal ( we can try to play on the morphological operators)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    # cv.imshow("sure_bg image", sure_bg)

    # sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    # cv.imshow("dist_transform image", dist_transform)
    ret, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # cv.imshow("sure_fg image", sure_fg)

    # unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    # cv.imshow("unknown image", unknown)

    # Marker labelling
    ret, markers = cv.connectedComponents(
        sure_fg)  # maybe markers with only sure_fg are less effective in the segmentation

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]
    cv.imshow("watershed image", image)

    cv.waitKey()
    cv.destroyAllWindows()
