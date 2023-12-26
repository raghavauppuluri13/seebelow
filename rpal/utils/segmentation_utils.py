from typing import Tuple
import numpy as np
import cv2 as cv
import argparse


def get_hsv_threshold(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Script to determine color range in HSV of a selected region
    :param img: image in BGR format
    :return lower and upper bound of color range
    """

    rois = []
    while True:
        r = cv.selectROI(img)
        if r == (0, 0, 0, 0):  # Check if 'Esc' was pressed
            break
        rois.append(r)
        cv.waitKey()

    # Initialize min and max arrays with extreme values
    min_vals = [255, 255, 255]
    max_vals = [0, 0, 0]

    for r in rois:
        im_crop = img[int(r[1]) : int(r[1] + r[3]), int(r[0]) : int(r[0] + r[2])]
        im_crop_hsv = cv.cvtColor(im_crop, cv.COLOR_BGR2HSV)

        for i in range(3):  # Iterate through each channel (H, S, V)
            min_vals[i] = min(min_vals[i], im_crop_hsv[..., i].min())
            max_vals[i] = max(max_vals[i], im_crop_hsv[..., i].max())

    lower_bound = np.array(min_vals)
    upper_bound = np.array(max_vals)

    print(
        f"Lower and upper bound of color range in cropped selection: {lower_bound}, {upper_bound}"
    )

    blur = cv.blur(img, (5, 5))
    blur0 = cv.medianBlur(blur, 5)
    blur1 = cv.GaussianBlur(blur0, (5, 5), 0)
    blur2 = cv.bilateralFilter(blur1, 9, 75, 75)

    cv.imshow("image with preprocessing", blur2)
    cv.waitKey()

    hsv = cv.cvtColor(blur2, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lower_bound, upper_bound)
    res = cv.bitwise_and(img, img, mask=mask)

    cv.imshow("image with mask", res)
    cv.waitKey()

    return lower_bound, upper_bound


def get_color_mask(
    frame: np.ndarray,
    threshold: Tuple[np.ndarray, np.ndarray],
    kernel_size=15,
    vis=False,
) -> np.ndarray:
    """Convert numpy RGB frame to HSV and and threshold image to given color range in HSV.
    Arguments:
    :param frame: image to generate mask in np array format
    :param threshold: hsv color range consisting of tuple of np arrays (lower color bound, upper color bound)
    :param kernel_size: kernel size for morphological operations
    :param vis: show image with mask using cv2.imshow
    :return: mask in np array format
    """

    frame

    blur = cv.blur(frame, (5, 5))
    blur0 = cv.medianBlur(blur, 5)
    blur1 = cv.GaussianBlur(blur0, (5, 5), 0)
    blur2 = cv.bilateralFilter(blur1, 9, 75, 75)
    hsvFrame = cv.cvtColor(blur2, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsvFrame, *threshold)

    # Morphological operations to remove small artifacts
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask_closed = cv.morphologyEx(mask_opened, cv.MORPH_CLOSE, kernel)
    out = cv.bitwise_and(frame, frame, mask=mask_closed)

    if vis:
        cv.imshow("image with mask", out)
        cv.waitKey()

    return mask_closed
