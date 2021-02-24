import cv2 as cv
import numpy as np

colors = []


def on_mouse_click(event, x, y, flags, frame):
    if event == cv.EVENT_LBUTTONUP:
        colors.append(frame[y, x].tolist())


if __name__ == "__main__":
    while True:
        image = cv.imread("../media/TR02 -  20200430.jpg")
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        if colors:
            cv.putText(hsv, str(colors[-1]), (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        cv.imshow('frame', hsv)
        cv.setMouseCallback('frame', on_mouse_click, hsv)

        if cv.waitKey(1000) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

    minh = min(c[0] for c in colors)
    mins = min(c[1] for c in colors)
    minv = min(c[2] for c in colors)
    maxh = max(c[0] for c in colors)
    maxs = max(c[1] for c in colors)
    maxv = max(c[2] for c in colors)
    lb = [minh, mins, minv]
    ub = [maxh, maxs, maxv]
    print(lb, ub)
