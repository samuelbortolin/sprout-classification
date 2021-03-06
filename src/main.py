from __future__ import absolute_import, annotations

from copy import deepcopy

import cv2 as cv
import numpy as np

from image_utils.image_opertions import StandardImageOperations as SIO


image_path = "../media/20200402_104001.jpg"  # in the future we can set the path as argument or env var


if __name__ == "__main__":

    image = cv.imread(image_path)  # in the future we can set the path as argument or env var
    image = SIO.rescale_image(image)
    greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # try a edge detector approach
    previous_threshold = SIO.canny_elements(greyscale_image, 25)
    previous_delta = SIO.canny_elements(greyscale_image, 24) - previous_threshold
    picked_threshold = 0
    for i in range(26, 1000):
        new_threshold = SIO.canny_elements(greyscale_image, i)
        delta = previous_threshold - new_threshold
        previous_threshold = new_threshold
        if (delta + previous_delta) < 50:
            picked_threshold = i
            break
        previous_delta = delta

    # show the original and the resulting image
    cv.imshow("original image", image)
    edges_on_image = SIO.canny_on_image(greyscale_image, picked_threshold, image)
    cv.imshow("relevant edges of the image", edges_on_image)

    # try to use area
    contour_image = SIO.canny_edges(greyscale_image, picked_threshold)
    contour_tuple = cv.findContours(contour_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = SIO.grab_contours(contour_tuple)

    for contour in contours:
        cv.drawContours(contour_image, [contour], 0, (100, 5, 10), 10)

    image_with_contours_area = deepcopy(image)
    for i, item_i in enumerate(contour_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                image_with_contours_area[i][j] = (0, 0, 0)
    cv.imshow("image with contours area", image_with_contours_area)

    # try color after edges
    color = input("What do you want to analyze? Write w for white/flowers, g for green/leaves or b for brown/branches: ")
    hsv_filtered_image = SIO.get_hsv_mask(image, cv.cvtColor(image, cv.COLOR_BGR2HSV), color)
    cv.imshow("hsv filtered image", hsv_filtered_image)
    edges_after_hsv = SIO.canny_on_image(cv.cvtColor(hsv_filtered_image, cv.COLOR_BGR2GRAY), picked_threshold, hsv_filtered_image)
    cv.imshow("relevant edges of the image after hsv", edges_after_hsv)
    cv.imshow("edges bitwise_and", cv.bitwise_and(edges_on_image, edges_after_hsv))

    # show original image and hsv filtered image with the edges marked with a red line
    image_and_edges = deepcopy(image)
    hsv_filtered_image_and_edges = deepcopy(hsv_filtered_image)
    for i, item_i in enumerate(edges_on_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                image_and_edges[i][j] = (0, 0, 255)
                hsv_filtered_image_and_edges[i][j] = (0, 0, 255)
    cv.imshow("edges on original image", image_and_edges)
    cv.imshow("edges on hsv filtered image", hsv_filtered_image_and_edges)

    image_and_edges_hsv = deepcopy(image)
    hsv_filtered_image_and_edges_hsv = deepcopy(hsv_filtered_image)
    for i, item_i in enumerate(cv.bitwise_and(edges_on_image, edges_after_hsv)):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                image_and_edges_hsv[i][j] = (0, 0, 255)
                hsv_filtered_image_and_edges_hsv[i][j] = (0, 0, 255)
    cv.imshow("edges after hsv on original image", image_and_edges_hsv)
    cv.imshow("edges after hsv on hsv filtered image", hsv_filtered_image_and_edges_hsv)

    # try a watershed segmentation approach
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
    bgr_threshold = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    for i, item_i in enumerate(edges_on_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                bgr_threshold[i][j] = (0, 0, 255)
    cv.imshow("edges on threshold image", bgr_threshold)

    # noise removal ( we can try to play on the morphological operators)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_background = cv.dilate(opening, kernel, iterations=3)
    # sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_foreground = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_foreground = np.uint8(sure_foreground)
    # unknown region
    unknown = cv.subtract(sure_background, sure_foreground)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_foreground)  # maybe markers with only sure_fg are less effective in the segmentation

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(image, markers)
    image[markers == -1] = [0, 0, 255]
    cv.imshow("watershed image", image)

    cv.waitKey()
    cv.destroyAllWindows()
