from __future__ import absolute_import, annotations

import cv2 as cv
import numpy as np


ratio = 3  # we can try to set a high threshold instead of using this ratio
kernel_size = 3  # we can try to understand what is that
image = cv.imread("../media/image.jpg")
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

def canny_threshold_with_image(low_threshold: int) -> None:
    blurred_image = cv.blur(greyscale_image, (3, 3))
    detected_edges = cv.Canny(blurred_image, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    edges_on_image = image * (mask[:, :, None].astype(image.dtype))
    cv.imshow("relevant edges of the image", edges_on_image)


if __name__ == "__main__":
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
    canny_threshold_with_image(picked_value)
    cv.waitKey()
