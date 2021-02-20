import cv2 as cv
import numpy


max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
img = cv.imread("../media/IMG_20190327_082544_01.JPG")
print(len(img[0]) * len(img))
scale = (len(img[0]) * len(img)/100000)**(1/2)
print(scale)
img = cv.resize(img, (0, 0), fx=(1/scale), fy=(1/scale))
greyscale_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(greyscale_img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)



def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(greyscale_img, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = img * (mask[:, :, None].astype(img.dtype))
    return numpy.sum(mask)

def CannyThresholdWithImage(val):
    low_threshold = val
    img_blur = cv.blur(greyscale_img, (3, 3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = img * (mask[:, :, None].astype(img.dtype))
    cv.imshow("img", dst)


if __name__ == "__main__":
    previousTreshold = CannyThreshold(25)
    previousDelta = CannyThreshold(24) - previousTreshold
    pickedValue = 0
    for i in range(26, 1000):
        newTreshold = CannyThreshold(i)
        delta = previousTreshold - newTreshold
        previousTreshold = newTreshold
        print(i, ": ", delta)
        if (delta + previousDelta) < 50:
            pickedValue = i
            print(CannyThreshold(i))
            break
        previousDelta = delta
    CannyThresholdWithImage(pickedValue)
    cv.imshow("original", img)
    cv.waitKey()


