# TODO
# convert HEIC to JPG: https://stackoverflow.com/questions/54395735/how-to-work-with-heic-image-file-types-in-python

# Links
# pytesseract: https://github.com/madmaze/pytesseract
# config option: https://stackoverflow.com/questions/44619077/pytesseract-ocr-multiple-config-options
# contour detection: https://learnopencv.com/contour-detection-using-opencv-python-c/
# contour detecion: https://stackoverflow.com/questions/57258173/opencv-join-contours-when-rectangle-overlaps-another-rect
# contour shapes and area: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
# https://stackoverflow.com/questions/55587820/how-to-get-the-only-min-area-rectangle-on-a-multiple-contours-image-with-cv2-min
# https://stackoverflow.com/questions/56829193/identifying-multiple-rectangles-and-draw-bounding-box-around-them-using-opencv
# https://michhar.github.io/masks_to_polygons_and_back/
# https://stackoverflow.com/questions/57282935/how-to-detect-area-of-pixels-with-the-same-color-using-opencv
# https://stackoverflow.com/questions/48477130/find-area-of-overlapping-rectangles-in-python-cv2-with-a-raw-list-of-points
# https://stackoverflow.com/questions/15424852/region-of-interest-opencv-python
# https://stackoverflow.com/questions/16538774/dealing-with-contours-and-bounding-rectangle-in-opencv-2-4-python-2-7
import pprint
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from collections import namedtuple
Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])

# https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles


def intersect_area(a, b):
    """Calcluate intersection area between two rectangles. Each rectangle has xmin, xmax, ymin, ymax fields."""
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)

    return float(dx*dy) if (dx >= 0) and (dy >= 0) else 0.


def normalize_images(images):
    """Convert all images into 3-dimensional images via cv2.COLOR_GRAY2BGR."""
    return [cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim == 2 else image for image in images]


def threshold_image(img_orig):
    """Grayscale image and apply Otsu's threshold"""
    # grayscale
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    # otsu's threshold
    img_thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return img_thresh


def mask_image(img, lowerb, upperb):
    # highlight colour range
    hsv_lower = [22, 30, 30]
    hsv_upper = [45, 255, 255]

    # rgb to HSV color spave conversion
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    HSV_lower = np.array(hsv_lower, np.uint8)  # Lower HSV value
    HSV_upper = np.array(hsv_upper, np.uint8)  # Upper HSV value
    # Threshold
    img_mask = cv2.inRange(img_hsv, HSV_lower, HSV_upper)
    # output = cv2.bitwise_and(img_orig, img_orig, mask=mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return img_mask


def draw_ocr_rectangles(img, data):
    n_boxes = len(data['level'])

    # draw rectangles for words
    for i in range(n_boxes):
        (x, y, w, h) = (data['left'][i], data['top']
                        [i], data['width'][i], data['height'][i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img


def main(args):
    img_input = str(args.img_input)

    # Load image
    img_orig = cv2.imread(img_input)

    # Grayscale and apply Otsu's threshold
    img_thresh = threshold_image(img_orig)

    # ocr
    data_ocr = pytesseract.image_to_data(
        img_thresh, lang='eng', config='--psm 11', output_type=Output.DICT)
    n_boxes = len(data_ocr['level'])

    # draw all ocr rect
    img_ocr = draw_ocr_rectangles(img_orig.copy(), data_ocr)

    # highlight colour range
    hsv_lowerb = [22, 30, 30]
    hsv_upperb = [45, 255, 255]

    img_mask = mask_image(img_orig, hsv_lowerb, hsv_upperb)

    # find connected components
    contours, hierarchy, = cv2.findContours(
        img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # draw only highlighted ocr rects
    img_ocr_highlight = img_orig.copy()
    for i in range(n_boxes):
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
                        [i], data_ocr['width'][i], data_ocr['height'][i])
        rect_word = Rectangle(x, y, x+w, y+h)
        area_word = float((rect_word.xmax - rect_word.xmin)
                          * (rect_word.ymax - rect_word.ymin))
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            rect_contour = Rectangle(x, y, x+w, y+h)
            area_intersect = intersect_area(rect_contour, rect_word)
            percent = area_intersect / area_word

            if (percent >= 0.5):
                cv2.rectangle(img_ocr_highlight, (rect_word.xmin, rect_word.ymin),
                              (rect_word.xmax, rect_word.ymax), (0, 255, 0), 2)

    img_mask_highlight = img_orig.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rect_contour = Rectangle(x, y, x+w, y+h)
        cv2.rectangle(img_mask_highlight, (rect_contour.xmin, rect_contour.ymin),
                      (rect_contour.xmax, rect_contour.ymax), (0, 255, 0), 2)

    cv2.drawContours(image=img_mask_highlight, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    img_mask_contour_filled2 = img_mask.copy()
    contours, hierarchy, = cv2.findContours(
        img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_mask_contour_filled2, contours, -1,
                     (255, 255, 255), cv2.FILLED, 8, hierarchy)

    img_mask_contour_filled3 = img_mask.copy()
    contours, hierarchy, = cv2.findContours(
        img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_mask_contour_filled3, contours, -1,
                     (255, 255, 255), cv2.FILLED, 8, hierarchy)

    box_w = 10
    box_h = 10
    threshold_perc = 25
    threshold = (box_w*box_h*threshold_perc)/100

    contours, hierarchy, = cv2.findContours(
        img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    img_mask_contour_filled = img_mask.copy()
    for idx, c in enumerate(contours):
        cv2.drawContours(img_mask_contour_filled, contours, idx,
                         (255, 255, 255), cv2.FILLED, 8, hierarchy)

        xmin, ymin, w, h = cv2.boundingRect(c)
        xmax = xmin + w
        ymax = ymin + h

        # Scan the image with in bounding box
        for x in range(xmin, xmax, box_w):
            for y in range(ymin, ymax, box_h):
                # Rect roi_rect(j, k, box_w, box_h)
                # Mat roi = dst(roi_rect)
                rect_roi = Rectangle(x, y, x+box_w, y+box_h)
                roi = img_mask_contour_filled[y:y+box_h, x:x+box_w]
                count = cv2.countNonZero(roi)

                if count > threshold:
                    cv2.rectangle(img_mask_highlight, (rect_roi.xmin, rect_roi.ymin),
                                  (rect_roi.xmax, rect_roi.ymax),
                                  (255, 0, 0), 2, 8, 0)

                    # contours, hierarchy, = cv2.findContours(
                    #     img_mask_contour_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                    # img_mask_contour_filled_rects = cv2.cvtColor(
                    #     img_mask_contour_filled.copy(),  cv2.COLOR_GRAY2BGR)
                    # for c in contours:
                    #     xmin, ymin, w, h = cv2.boundingRect(c)
                    #     xmax = xmin + w
                    #     ymax = ymin + h

                    #     # Scan the image with in bounding box
                    #     for x in range(xmin, xmax, box_w):
                    #         for y in range(ymin, ymax, box_h):
                    #             # Rect roi_rect(j, k, box_w, box_h)
                    #             # Mat roi = dst(roi_rect)
                    #             rect_roi = Rectangle(x, y, x+box_w, y+box_h)
                    #             roi = img_mask_contour_filled[y:y+box_h, x:x+box_w]
                    #             count = cv2.countNonZero(roi)

                    #             # cv2.rectangle(img_mask_contour_filled_rects, (rect_roi.xmin, rect_roi.ymin),
                    #             #               (rect_roi.xmax, rect_roi.ymax),
                    #             #               (255, 0, 0), cv2.FILLED, 8, 0)
                    #             if count > threshold:
                    #                 cv2.rectangle(img_mask_contour_filled_rects, (rect_roi.xmin, rect_roi.ymin),
                    #                               (rect_roi.xmax, rect_roi.ymax),
                    #                               (255, 0, 0), cv2.FILLED, 8, 0)

                    # stack images
    img_ocr_row = np.concatenate(normalize_images(
        (
            img_orig,
            img_orig,
            img_orig,
            img_thresh,
            img_ocr
        )), axis=1)

    # img_ocr_row = vstack((img_orig, img_thresh, img_ocr))

    img_contour_row = np.concatenate(normalize_images(
        (
            img_mask,
            img_mask_highlight,
            img_mask_contour_filled,
            img_mask_contour_filled2,
            img_mask_contour_filled2,
            # img_mask_contour_filled_rects,
            # img_ocr_highlight
        )), axis=1)

    img_grid = np.concatenate((img_ocr_row, img_contour_row), axis=0)

    cv2.imshow('img', img_grid)

    cv2.waitKey(0)


if __name__ == "__main__":

    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('img_input', type=Path, help="Input image")
    args = parser.parse_args()
    main(args)
