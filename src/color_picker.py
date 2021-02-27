import cv2 as cv
import numpy as np

from play_with_HSV import rescale_image


colors = []


def on_mouse_click(event, x, y, flags, hsv_image):
    # mouse click function to store the HSV value

    if event == cv.EVENT_LBUTTONUP:
        colors.append(hsv_image[y, x].tolist())


def callback(x):
    # trackbar callback function to update HSV value

    global H_low, H_high, S_low, S_high, V_low, V_high
    # assign trackbar position value to H, S, V high and low variable
    H_low = cv.getTrackbarPos("low_H", "control_H")
    H_high = cv.getTrackbarPos("high_H", "control_H")

    S_low = cv.getTrackbarPos("low_S", "control_S")
    S_high = cv.getTrackbarPos("high_S", "control_S")

    V_low = cv.getTrackbarPos("low_V", "control_V")
    V_high = cv.getTrackbarPos("high_V", "control_V")


if __name__ == "__main__":

    image_path = "../media/image.jpg"

    while True:
        image = cv.imread(image_path)
        image = rescale_image(image)
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
    # masking HSV value selected color becomes black
    result_image = cv.bitwise_and(image, image, mask=mask)

    # show image
    cv.imshow("mask", mask)
    cv.imshow("result image", result_image)

    # wait for the user to press escape and break the while loop
    cv.waitKey()
    cv.destroyAllWindows()

    # global variables
    H_low = 0
    H_high = 179
    S_low = 0
    S_high = 255
    V_low = 0
    V_high = 255

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

    while True:
        # read source image and convert to HSV color
        image = cv.imread(image_path)
        image = rescale_image(image)
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        lower_bound = np.array([H_low, S_low, V_low])
        upper_bound = np.array([H_high, S_high, V_high])

        mask = cv.inRange(hsv_image, lower_bound, upper_bound)
        result_image = cv.bitwise_and(image, image, mask=mask)

        cv.imshow("mask", mask)
        cv.imshow("result image", result_image)

        if cv.waitKey() & 0xFF == ord("q"):
            break

    cv.destroyAllWindows()
    print(lower_bound, upper_bound)
