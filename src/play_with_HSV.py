import cv2 as cv
import numpy as np


def apply_mask(original_image: np.ndarray, hsv_image: np.ndarray, lower_bound: np.ndarray, upper_bound: np.ndarray) -> np.ndarray:
    mask = cv.inRange(hsv_image, lower_bound, upper_bound)
    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, (5, 5), iterations=1)
    return cv.bitwise_and(original_image, original_image, mask=mask)


def rescale_image(image_to_rescale: np.ndarray, target_number_of_pixels: int = 100000) -> np.ndarray:
    scale = (len(image_to_rescale[0]) * len(image_to_rescale) / target_number_of_pixels) ** (1 / 2)
    # rescale image all to the same standard number of pixels
    return cv.resize(image_to_rescale, (0, 0), fx=(1 / scale), fy=(1 / scale))


if __name__ == "__main__":
    # x flowers
    image_path = "../media/ROBI_VT01_20200507_3.png"
    image = cv.imread(image_path)
    image = rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_bound = np.array([15, 0, 100])
    upper_bound = np.array([35, 40, 255])
    cv.imshow("result after applying hsv mask", apply_mask(image, hsv_image, lower_bound, upper_bound))
    cv.waitKey()
    cv.destroyAllWindows()

    # x leaves
    image_path = "../media/FOTO1ROBI_MI02_20200422.jpg"
    image = cv.imread(image_path)
    image = rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_bound = np.array([20, 0, 0])
    upper_bound = np.array([80, 255, 255])
    cv.imshow("result after applying hsv mask", apply_mask(image, hsv_image, lower_bound, upper_bound))
    cv.waitKey()
    cv.destroyAllWindows()

    # x branches
    image_path = "../media/image.jpg"
    image = cv.imread(image_path)
    image = rescale_image(image)
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([30, 255, 255])
    cv.imshow("result after applying hsv mask", apply_mask(image, hsv_image, lower_bound, upper_bound))
    cv.waitKey()
    cv.destroyAllWindows()
