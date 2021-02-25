import cv2 as cv
import numpy as np


def apply_mask(original_image: np.ndarray, hsv_image: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> None:
    mask = cv.inRange(hsv_image, lower_bound, upper_bound)
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, (5, 5), iterations=1)
    # mask = cv.dilate(mask, None, iterations=2)
    # res = cv.bitwise_and(frame_hsv, frame_hsv, mask=mask)  # maybe use this instead of go through all the image

    for i, item_i in enumerate(mask):
        for j, item_j in enumerate(item_i):
            if item_j.all() == 0:
                original_image[i][j] = (0, 0, 0)

    cv.imshow("result after applying hsv mask", original_image)
    cv.waitKey()


def rescale_image(image_to_rescale: np.ndarray, target_number_of_pixels: int = 100000) -> np.ndarray:
    scale = (len(image_to_rescale[0]) * len(image_to_rescale) / target_number_of_pixels) ** (1 / 2)
    # rescale image all to the same standard number of pixels
    return cv.resize(image_to_rescale, (0, 0), fx=(1 / scale), fy=(1 / scale))


if __name__ == "__main__":
    # x flowers
    image = cv.imread("../media/TR02 -  20200430.jpg")
    image = rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_bound = np.array([15, 0, 100])
    upper_bound = np.array([35, 40, 255])
    apply_mask(image, hsv_image, lower_bound, upper_bound)

    # x leaves
    image = cv.imread("../media/ROb_BBCH51_poggi.jpg")
    image = rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_bound = np.array([35, 100, 100])
    upper_bound = np.array([80, 255, 255])
    apply_mask(image, hsv_image, lower_bound, upper_bound)

    # x branches
    image = cv.imread("../media/image.jpg")
    image = rescale_image(image)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_bound = np.array([30, 0, 0])
    upper_bound = np.array([50, 75, 200])
    apply_mask(image, hsv_image, lower_bound, upper_bound)

    cv.destroyAllWindows()
