from __future__ import absolute_import, annotations

from copy import deepcopy

import cv2 as cv
import numpy as np


ratio = 3  # we can try to set a high threshold instead of using this ratio
kernel_size = 3  # we can try to understand what is that
image = cv.imread("../media/image.jpg")  # in the future we can set the path as argument or env var
# print(len(img[0]) * len(img))
scale = (len(image[0]) * len(image) / 100000) ** (1 / 2)
# print(scale)

# rescale image all to the same standard number of pixels
image = cv.resize(image, (0, 0), fx=(1 / scale), fy=(1 / scale))
greyscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def canny_threshold(low_threshold: int) -> int:
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
    previous_threshold = canny_threshold(25)
    previous_delta = canny_threshold(24) - previous_threshold
    picked_value = 0
    for i in range(26, 1000):
        new_threshold = canny_threshold(i)
        delta = previous_threshold - new_threshold
        previous_threshold = new_threshold
        # print(i, ": ", delta)
        if (delta + previous_delta) < 50:
            picked_value = i
            # print(canny_threshold(i))
            break
        previous_delta = delta

    # show the original and the resulting image
    cv.imshow("original image", image)
    edges_on_image = canny_threshold_with_image(picked_value)
    cv.imshow("relevant edges of the image", edges_on_image)

    image_and_edges = deepcopy(image)
    for i, item_i in enumerate(edges_on_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                image_and_edges[i][j] = (0, 0, 255)
    cv.imshow("edges on original image", image_and_edges)

    # try a segmentation approach
    ret, thresh = cv.threshold(greyscale_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    cv.imshow("thresh image", thresh)

    bgr_thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
    for i, item_i in enumerate(edges_on_image):
        for j, item_j in enumerate(item_i):
            if item_j.all() != 0:
                bgr_thresh[i][j] = (0, 0, 255)
    cv.imshow("edges on threshold image", bgr_thresh)

    # noise removal ( we can tryy to play on the morphological operators)
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
    ret, markers = cv.connectedComponents(sure_fg)  # maybe markers with only sure_fg are less effective in the segmentation

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]

    cv.imshow("watershed image", image)
    cv.waitKey()
