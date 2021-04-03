from __future__ import absolute_import, annotations

import cv2 as cv
import numpy as np

from image_utils.standard_image_operations import StandardImageOperations as SIO


image_path = "../images/image.extension"

colors = []

H_low = 0
H_high = 179
S_low = 0
S_high = 255
V_low = 0
V_high = 255


def on_mouse_click(event, x, y, flags, hsv_image):
    # mouse click function to store the hsv value

    if event == cv.EVENT_LBUTTONUP:
        colors.append(hsv_image[y, x].tolist())


def callback(x):
    # trackbar callback function assigning trackbar position value to H, S, V high and low variables

    global H_low, H_high, S_low, S_high, V_low, V_high
    H_low = cv.getTrackbarPos("low_H", "control_H")
    H_high = cv.getTrackbarPos("high_H", "control_H")
    S_low = cv.getTrackbarPos("low_S", "control_S")
    S_high = cv.getTrackbarPos("high_S", "control_S")
    V_low = cv.getTrackbarPos("low_V", "control_V")
    V_high = cv.getTrackbarPos("high_V", "control_V")


if __name__ == "__main__":

    # selects points to build a mask
    while True:
        image = cv.imread(image_path)
        image = SIO.rescale_image(image)
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        if colors:
            cv.putText(hsv_image, str(colors[-1]), (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv.imshow("hsv image", hsv_image)
        cv.setMouseCallback("hsv image", on_mouse_click, hsv_image)

        if cv.waitKey(100) & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()

    min_h = min(c[0] for c in colors)
    min_s = min(c[1] for c in colors)
    min_v = min(c[2] for c in colors)
    max_h = max(c[0] for c in colors)
    max_s = max(c[1] for c in colors)
    max_v = max(c[2] for c in colors)

    lower_bound = np.array([min_h, min_s, min_v])
    upper_bound = np.array([max_h, max_s, max_v])
    print(lower_bound, upper_bound)

    mask = cv.inRange(hsv_image, lower_bound, upper_bound)
    result_image = cv.bitwise_and(image, image, mask=mask)
    cv.imshow("mask", mask)
    cv.imshow("result image", result_image)

    # show complementary on h
    lower_bound = np.array([0, min_s, min_v])
    upper_bound = np.array([min_h, max_s, max_v])
    mask_lower = cv.inRange(hsv_image, lower_bound, upper_bound)
    result_lower_image = cv.bitwise_and(image, image, mask=mask_lower)

    lower_bound = np.array([max_h, min_s, min_v])
    upper_bound = np.array([179, max_s, max_v])
    mask_upper = cv.inRange(hsv_image, lower_bound, upper_bound)
    result_upper_image = cv.bitwise_and(image, image, mask=mask_upper)

    complementary_h_mask = np.bitwise_or(mask_lower, mask_upper)
    result_complementary_h_image = cv.bitwise_and(image, image, mask=complementary_h_mask)
    cv.imshow("complementary H mask", complementary_h_mask)
    cv.imshow("result complementary H image", result_complementary_h_image)

    cv.waitKey()
    cv.destroyAllWindows()

    # create separate windows and trackbars for high,low H,S,V
    cv.namedWindow("control_H", 2)
    cv.createTrackbar("low_H", "control_H", 0, 179, callback)
    cv.createTrackbar("high_H", "control_H", 179, 179, callback)
    cv.resizeWindow("control_H", 400, 20)

    cv.namedWindow("control_S", 2)
    cv.createTrackbar("low_S", "control_S", 0, 255, callback)
    cv.createTrackbar("high_S", "control_S", 255, 255, callback)
    cv.resizeWindow("control_S", 400, 20)

    cv.namedWindow("control_V", 2)
    cv.createTrackbar("low_V", "control_V", 0, 255, callback)
    cv.createTrackbar("high_V", "control_V", 255, 255, callback)
    cv.resizeWindow("control_V", 400, 20)

    # move trackbars to build a mask
    while True:
        image = cv.imread(image_path)
        image = SIO.rescale_image(image)
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        lower_bound = np.array([H_low, S_low, V_low])
        upper_bound = np.array([H_high, S_high, V_high])

        mask = cv.inRange(hsv_image, lower_bound, upper_bound)
        result_image = cv.bitwise_and(image, image, mask=mask)
        cv.imshow("hsv mask", mask)
        cv.imshow("hsv filtered image", result_image)

        if cv.waitKey() & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()
    print(lower_bound, upper_bound)
