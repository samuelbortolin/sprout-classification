from __future__ import absolute_import, annotations

from copy import deepcopy

import cv2 as cv
import numpy as np

from image_utils.image_opertions import StandardImageOperations as SIO


image_path = "../images/image.extension"


if __name__ == "__main__":

    image = cv.imread(image_path)
    image = SIO.rescale_image(image)
    cv.imshow("original image", image)
    greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # try an edge detector approach
    sobel_edges = SIO.sobel_edges(greyscale_image)
    sobel_edges_after_percentile_removal = SIO.remove_percentile(sobel_edges)
    canny_edges = SIO.canny_edges(greyscale_image, SIO.find_canny_best_threshold(greyscale_image))

    canny_bitwise_and_sobel_after_percentile_removal = np.zeros((len(image), len(image[0])))
    for i, item_i in enumerate(canny_edges):
        for j, item_j in enumerate(item_i):
            if item_j.any() > 0 and sobel_edges_after_percentile_removal[i, j] > 0:
                canny_bitwise_and_sobel_after_percentile_removal[i, j] = int(255)
            else:
                canny_bitwise_and_sobel_after_percentile_removal[i, j] = int(0)
    canny_bitwise_and_sobel_after_percentile_removal = canny_bitwise_and_sobel_after_percentile_removal.astype(np.uint8)
    cv.imshow("canny bitwise_and sobel after percentile removal", canny_bitwise_and_sobel_after_percentile_removal)

    image_and_edges = deepcopy(image)
    for i, item_i in enumerate(canny_bitwise_and_sobel_after_percentile_removal):
        for j, item_j in enumerate(item_i):
            if item_j != 0:
                image_and_edges[i][j] = (0, 0, 255)
    cv.imshow("edges on original image", image_and_edges)

    # try to use area
    contour_tuple = cv.findContours(canny_bitwise_and_sobel_after_percentile_removal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = SIO.grab_contours(contour_tuple)
    contour_image = deepcopy(canny_bitwise_and_sobel_after_percentile_removal)

    for contour in contours:
        cv.drawContours(contour_image, [contour], 0, (100, 5, 10), 10)

    image_with_contours_area = deepcopy(image)
    for i, item_i in enumerate(contour_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                image_with_contours_area[i][j] = (0, 0, 0)
    cv.imshow("image with contours area", image_with_contours_area)

    # try hsv color filter after edges
    color = input("What do you want to analyze? Type f for flowers, l for leaves or b for branches: ")
    hsv_filtered_image = SIO.get_hsv_mask(image, cv.cvtColor(image, cv.COLOR_BGR2HSV), color)
    cv.imshow("hsv filtered image", hsv_filtered_image)

    hsv_edges = SIO.canny_edges(cv.cvtColor(hsv_filtered_image, cv.COLOR_BGR2GRAY), SIO.find_canny_best_threshold(cv.cvtColor(hsv_filtered_image, cv.COLOR_BGR2GRAY)))
    hsv_bitwise_and_canny_and_sobel = np.zeros((len(image), len(image[0])))
    for i, item_i in enumerate(cv.bitwise_and(canny_edges, hsv_edges)):
        for j, item_j in enumerate(item_i):
            if item_j.any() > 0 and sobel_edges_after_percentile_removal[i, j] > 0:
                hsv_bitwise_and_canny_and_sobel[i, j] = int(255)
            else:
                hsv_bitwise_and_canny_and_sobel[i, j] = int(0)
    hsv_bitwise_and_canny_and_sobel = hsv_bitwise_and_canny_and_sobel.astype(np.uint8)
    cv.imshow("hsv bitwise_and sobel-canny", hsv_bitwise_and_canny_and_sobel)

    image_and_edges_hsv = deepcopy(image)
    hsv_filtered_image_and_edges_hsv = deepcopy(hsv_filtered_image)
    for i, item_i in enumerate(hsv_bitwise_and_canny_and_sobel):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                image_and_edges_hsv[i][j] = (0, 0, 255)
                hsv_filtered_image_and_edges_hsv[i][j] = (0, 0, 255)
    cv.imshow("edges after hsv on original image", image_and_edges_hsv)
    cv.imshow("edges after hsv on hsv filtered image", hsv_filtered_image_and_edges_hsv)

    cv.waitKey()
    cv.destroyAllWindows()
