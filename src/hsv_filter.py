from __future__ import absolute_import, annotations

import cv2 as cv

from image_utils.image_opertions import StandardImageOperations as SIO


flowers_image_path = "../images/flowers_image.extension"
leaves_image_path = "../images/leaves_image.extension"
branches_image_path = "../images/branches_image.extension"


if __name__ == "__main__":

    # x flowers
    image = cv.imread(flowers_image_path)
    image = SIO.rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("result after applying hsv mask to image with flowers", SIO.get_hsv_mask(image, hsv_image, "w"))
    cv.waitKey()
    cv.destroyAllWindows()

    # x leaves
    image = cv.imread(leaves_image_path)
    image = SIO.rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("result after applying hsv mask to image with leaves", SIO.get_hsv_mask(image, hsv_image, "g"))
    cv.waitKey()
    cv.destroyAllWindows()

    # x branches
    image = cv.imread(branches_image_path)
    image = SIO.rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("result after applying hsv mask to image with branches", SIO.get_hsv_mask(image, hsv_image, "b"))
    cv.waitKey()
    cv.destroyAllWindows()
