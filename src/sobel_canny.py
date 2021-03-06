from __future__ import absolute_import, annotations

from copy import deepcopy

import cv2 as cv
import numpy as np

from image_utils.image_opertions import StandardImageOperations as SIO


image_path = "../media/20200402_104028_001.jpg"  # in the future we can set the path as argument or env var

percentage = 85
scale = 1
delta = 0
depth = cv.CV_16S
n_derivative = 2


if __name__ == "__main__":

    image = cv.imread(image_path)
    image = SIO.rescale_image(image)
    color = input("What do you want to analyze? Write w for white/flowers, g for green/leaves or b for brown/branches: ")
    hsv_filtered_image = SIO.get_hsv_mask(image, cv.cvtColor(image, cv.COLOR_BGR2HSV), color)
    greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(image, (3, 3), 0)
    greyscale_blurred_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)

    # try with sobel
    gradient_x = cv.Sobel(greyscale_blurred_image, depth, n_derivative, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    gradient_y = cv.Sobel(greyscale_blurred_image, depth, 0, n_derivative, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    abs_gradient_x = cv.convertScaleAbs(gradient_x)
    abs_gradient_y = cv.convertScaleAbs(gradient_y)
    gradient_image = cv.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)
    cv.imshow("sobel gradient image", gradient_image)
    kernel = np.ones((3, 3), np.uint8)
    close_after_sobel = cv.morphologyEx(gradient_image, cv.MORPH_CLOSE, kernel)
    cv.imshow("close after sobel", close_after_sobel)
    open_after_close_on_sobel = cv.morphologyEx(close_after_sobel, cv.MORPH_OPEN, kernel)
    cv.imshow("open after close on sobel", open_after_close_on_sobel)

    # try to remove a percentile
    positive_values_array = []
    for i, item_i in enumerate(open_after_close_on_sobel):
        for j, item_j in enumerate(item_i):
            if item_j > 0:
                positive_values_array.append(item_j)
    percentile = np.percentile(positive_values_array, percentage)

    for i, item_i in enumerate(open_after_close_on_sobel):
        for j, item_j in enumerate(item_i):
            if item_j <= percentile:
                open_after_close_on_sobel[i, j] = 0
    cv.imshow("percentile removed", open_after_close_on_sobel)

    # try a canny edge detector approach
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

    edged_image = SIO.canny_edges(greyscale_image, picked_threshold)
    bitwise_and_percentile_canny = np.zeros((len(image), len(image[0])))
    for i, item_i in enumerate(edged_image):
        for j, item_j in enumerate(item_i):
            if item_j.any() > 0 and open_after_close_on_sobel[i, j] > 0:
                bitwise_and_percentile_canny[i, j] = int(255)
            else:
                bitwise_and_percentile_canny[i, j] = int(0)
    bitwise_and_percentile_canny = bitwise_and_percentile_canny.astype(np.uint8)
    cv.imshow("bitwise_and percentile canny", bitwise_and_percentile_canny)

    # try to use also hsv
    bitwise_and_on_image_hsv = deepcopy(hsv_filtered_image)
    for i, item_i in enumerate(bitwise_and_percentile_canny):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                bitwise_and_on_image_hsv[i][j] = (0, 0, 255)
    cv.imshow("bitwise_and percentile canny on image hsv", bitwise_and_on_image_hsv)

    # try to use area
    contour_tuple = cv.findContours(bitwise_and_percentile_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = SIO.grab_contours(contour_tuple)
    contour_image = deepcopy(bitwise_and_percentile_canny)

    for contour in contours:
        cv.drawContours(contour_image, [contour], 0, (100, 5, 10), 10)

    image_with_contours_area = deepcopy(image)
    for i, item_i in enumerate(contour_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                image_with_contours_area[i][j] = (0, 0, 0)
    cv.imshow("image with contours area", image_with_contours_area)

    # store images on file
    cv.imwrite("../test_images/original_image_0.jpg", image)
    cv.imwrite("../test_images/bitwise_canny_percentile_0.jpg", bitwise_and_percentile_canny)
    cv.imwrite("../test_images/hsv_image_0.jpg", hsv_filtered_image)
    cv.imwrite("../test_images/bitwise_edges_on_image_0.jpg", bitwise_and_on_image_hsv)
    cv.imwrite("../test_images/bitwise_canny_percentile_area_0.jpg", image_with_contours_area)

    cv.waitKey()
    cv.destroyAllWindows()
