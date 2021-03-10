from __future__ import absolute_import, annotations

import cv2 as cv

from image_utils.image_opertions import StandardImageOperations as SIO


image_flowers_path = "../media/ROBI_VT01_20200507_3.png"
image_leaves_path = "../media/FOTO1ROBI_MI02_20200422.jpg"
image_branches_path = "../media/image.jpg"


if __name__ == "__main__":

    # x flowers
    image = cv.imread(image_flowers_path)
    image = SIO.rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("result after applying hsv mask", SIO.get_hsv_mask(image, hsv_image, "w"))
    cv.waitKey()
    cv.destroyAllWindows()

    # x leaves
    image = cv.imread(image_leaves_path)
    image = SIO.rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("result after applying hsv mask", SIO.get_hsv_mask(image, hsv_image, "g"))
    cv.waitKey()
    cv.destroyAllWindows()

    # x branches
    image = cv.imread(image_branches_path)
    image = SIO.rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("result after applying hsv mask", SIO.get_hsv_mask(image, hsv_image, "b"))
    cv.waitKey()
    cv.destroyAllWindows()
