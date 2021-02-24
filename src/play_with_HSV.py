import cv2 as cv
import numpy as np


if __name__ == "__main__":
    # x flowers
    image = cv.imread("../media/TR02 -  20200430.jpg")
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    l_b = np.array([15, 0, 100])
    u_b = np.array([35, 40, 255])

    mask = cv.inRange(hsv, l_b, u_b)

    for i, item_i in enumerate(mask):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                image[i][j] = (0, 0, 0)
    cv.imshow("result", image)

    cv.waitKey()
    cv.destroyAllWindows()

    # x leaves
    image = cv.imread("../media/ROb_BBCH51_poggi.jpg")
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    l_b = np.array([35, 100, 100])
    u_b = np.array([80, 255, 255])

    mask = cv.inRange(hsv, l_b, u_b)

    for i, item_i in enumerate(mask):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                image[i][j] = (0, 0, 0)
    cv.imshow("result", image)

    cv.waitKey()
    cv.destroyAllWindows()

    # x branches
    image = cv.imread("../media/image.jpg")
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    l_b = np.array([30, 0, 0])
    u_b = np.array([50, 75, 200])

    mask = cv.inRange(hsv, l_b, u_b)

    for i, item_i in enumerate(mask):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                image[i][j] = (0, 0, 0)
    cv.imshow("result", image)

    cv.waitKey()
    cv.destroyAllWindows()
