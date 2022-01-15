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
# https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
# https://www.freedomvc.com/index.php/2021/07/05/contours-and-hierarchy/
# https://www.pyimagesearch.com/2014/05/19/building-pokedex-python-comparing-shape-descriptors-opencv/
# https://stackoverflow.com/a/54734716/1967693
# https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
# https://answers.opencv.org/question/25912/split-contours-into-many-small-rectangles/
# https://stackoverflow.com/questions/69214202/using-pytesseract-to-get-text-from-an-image
# https://stackoverflow.com/questions/60110313/opencv-thresholding-adaptive-to-different-lightning-conditions
# https://www.pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/
# https://stackoverflow.com/questions/68107172/opencv-output-of-adaptive-threshold
# https://stackoverflow.com/questions/61461520/does-anyone-knows-the-meaning-of-output-of-image-to-data-image-to-osd-methods-o
# https://medium.com/geekculture/tesseract-ocr-understanding-the-contents-of-documents-beyond-their-text-a98704b7c655
# https://www.opcito.com/blogs/extracting-text-from-images-with-tesseract-ocr-opencv-and-python
# https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
import pprint
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from collections import namedtuple
import itertools
import re

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
Word = namedtuple('Word', ['level', 'page_num',
                  'block_num', 'par_num', 'line_num', 'word_num', 'top', 'left', 'width', 'height', 'conf', 'text'])

# https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles


class Levels:
    PAGE = 1
    BLOCK = 2
    PARAGRAPH = 3
    LINE = 4
    WORD = 5


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
    # Grayscale
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    # Binary inverse and Otsu's threshold
    img_thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return img_thresh


def mask_image(img_src, lower, upper):
    """Convert image from RGB to HSV and create a mask for given lower and upper boundaries."""
    # RGB to HSV color space conversion
    img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(lower, np.uint8)  # Lower HSV value
    hsv_upper = np.array(upper, np.uint8)  # Upper HSV value

    # Color segmentation with lower and upper threshold ranges to obtain a binary image
    img_mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
    # output = cv2.bitwise_and(img_src, img_src, mask=img_mask)

    # Morphological transformations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_mask = cv2.morphologyEx(
        img_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find highligted contour and fill them with white color
    contours, hierarchy, = cv2.findContours(
        img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy, = cv2.findContours(
    #     img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for idx, c in enumerate(contours):
        # if(hierarchy[0][idx][3] != -1):  # Discard contours that are holes
        #     continue
        cv2.drawContours(img_mask, contours, idx,
                         (255, 255, 255), cv2.FILLED, 8, hierarchy)

    return img_mask, contours, hierarchy


def mark_all_words(img_result, data_ocr):

    # draw rectangles for words
    for i in range(len(data_ocr['text'])):
        if data_ocr['level'][i] != Levels.WORD:
            continue
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
                        [i], data_ocr['width'][i], data_ocr['height'][i])
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_result


# def draw_contour_ocr_rectangles(img_mask, img_result, data):
#     # find connected components
#     contours, hierarchy, = cv2.findContours(
#         img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     n_boxes = len(data['level'])

#     for i in range(n_boxes):
#         (x, y, w, h) = (data['left'][i], data['top']
#                         [i], data['width'][i], data['height'][i])
#         rect_word = Rectangle(x, y, x+w, y+h)
#         area_word = float((rect_word.xmax - rect_word.xmin)
#                           * (rect_word.ymax - rect_word.ymin))
#         for c in contours:
#             x, y, w, h = cv2.boundingRect(c)
#             rect_contour = Rectangle(x, y, x+w, y+h)
#             area_intersect = intersect_area(rect_contour, rect_word)
#             percent = area_intersect / area_word

#             if (percent >= 0.5):
#                 cv2.rectangle(img_result, (rect_word.xmin, rect_word.ymin),
#                               (rect_word.xmax, rect_word.ymax), (0, 255, 0), 2)

#     return img_result


def draw_contour_rectangles(img_contour, img_result, contours, rect_width=10, rect_height=10, threshold_percentage=25):
    """Draw small rectangles within the contour if the respective rectangle area exceeds the defined threshold percentage."""

    # threshold for rectangle area
    threshold = (rect_width * rect_height * threshold_percentage) / 100
    # contours, hierarchy, = cv2.findContours(
    #     img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy, = cv2.findContours(
    #     img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, c in enumerate(contours):
        # if(hierarchy[0][idx][3] != -1):  # Discard contours that are holes
        #     continue
        # cv2.drawContours(img_mask, contours, idx,
        #                  (255, 255, 255), cv2.FILLED, 8, hierarchy)
        xmin, ymin, w, h = cv2.boundingRect(c)
        xmax = xmin + w
        ymax = ymin + h

        # Scan the image with in bounding boxes
        for x in range(xmin, xmax, rect_width):
            for y in range(ymin, ymax, rect_height):
                rect_roi = Rectangle(x, y, x+rect_width, y+rect_height)
                img_roi = img_contour[y:y+rect_height, x:x+rect_width]

                # count white pixels within region of interest
                count = cv2.countNonZero(img_roi)

                if count > threshold:
                    cv2.rectangle(img_result, (rect_roi.xmin, rect_roi.ymin),
                                  (rect_roi.xmax, rect_roi.ymax),
                                  (255, 0, 0), 1, 8, 0)

    return img_result


# def extract_words(img_mask, data_ocr, threshold_percentage=25) -> list[Word]:
#     extracted_words = []

#     for i in range(len(data_ocr['text'])):
#         (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
#                         [i], data_ocr['width'][i], data_ocr['height'][i])
#         # rect_roi = Rectangle(x, y, x+w, y+h)
#         rect_threshold = (w * h * threshold_percentage) / 100
#         img_roi = img_mask[y:y+h, x:x+w]
#         count = cv2.countNonZero(img_roi)

#         if count > rect_threshold:
#             word = Word(data_ocr['level'][i], data_ocr['page_num'][i], data_ocr['block_num'][i], data_ocr['par_num'][i], data_ocr['line_num'][i],
#                         data_ocr['word_num'][i], data_ocr['top'][i], data_ocr['left'][i], data_ocr['width'][i], data_ocr['height'][i], data_ocr['conf'][i], data_ocr['text'][i])
#             extracted_words.append(word)

#             print("Level: {}; Page: {}; Block: {}; Paragraph: {}; Line: {}; Word: {}; Text: {}".format(
#                 word.level, word.page_num, word.block_num, word.par_num, word.line_num, word.word_num, word.text))

#     return extracted_words

def find_highlighted_words(img_mask, data_ocr, threshold_percentage=25):
    # initiliaze new column with false values
    data_ocr['highlighted'] = [False] * len(data_ocr['text'])

    for i in range(len(data_ocr['text'])):
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
                        [i], data_ocr['width'][i], data_ocr['height'][i])
        # rect_roi = Rectangle(x, y, x+w, y+h)
        rect_threshold = (w * h * threshold_percentage) / 100
        img_roi = img_mask[y:y+h, x:x+w]
        count = cv2.countNonZero(img_roi)

        if count > rect_threshold:
            data_ocr['highlighted'][i] = True

    return data_ocr


def mark_highlighted_words(img_result, data_ocr):
    # n_boxes = len(data['text'])
    # contours, hierarchy, = cv2.findContours(
    #     img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw rectangles for words
    for i in range(len(data_ocr['text'])):
        if data_ocr['level'][i] != Levels.WORD:
            continue
        if not data_ocr['highlighted'][i]:
            continue

        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
                        [i], data_ocr['width'][i], data_ocr['height'][i])
        rect_roi = Rectangle(x, y, x+w, y+h)

        cv2.rectangle(img_result, (rect_roi.xmin, rect_roi.ymin),
                      (rect_roi.xmax, rect_roi.ymax), (0, 255, 0), 2)
    # for i in range(len(data['text'])):
    #     (x, y, w, h) = (data['left'][i], data['top']
    #                     [i], data['width'][i], data['height'][i])
    #     rect_roi = Rectangle(x, y, x+w, y+h)
    #     rect_threshold = (w * h * threshold_percentage) / 100

    #     img_roi = img_contour[y:y+h, x:x+w]
    #     count = cv2.countNonZero(img_roi)

    #     if count > rect_threshold:
    #         print("Level: {}; Page: {}; Block: {}; Paragraph: {}; Line: {}; Word: {}; Text: {}".format(
    #             data['level'][i],
    #             data['page_num'][i],
    #             data['block_num'][i],
    #             data['par_num'][i],
    #             data['line_num'][i],
    #             data['word_num'][i],
    #             data['text'][i]))
    #         cv2.rectangle(img_result, (rect_roi.xmin, rect_roi.ymin),
    #                       (rect_roi.xmax, rect_roi.ymax), (0, 255, 0), 2)

    return img_result


def words_to_string(data_ocr):

    word_list = []
    line_breaks = (Levels.PAGE, Levels.BLOCK, Levels.PARAGRAPH, Levels.LINE)

    for i in range(len(data_ocr['text'])):
        print("Level: {}; Page: {}; Block: {}; Paragraph: {}; Line: {}; Word: {}; Highlighted: {} Text: {}".format(
            data_ocr['level'][i],
            data_ocr['page_num'][i],
            data_ocr['block_num'][i],
            data_ocr['par_num'][i],
            data_ocr['line_num'][i],
            data_ocr['word_num'][i],
            data_ocr['highlighted'][i],
            data_ocr['text'][i]))

        if data_ocr['level'][i] in line_breaks:
            word_list.append("\n")
            continue

        text = data_ocr['text'][i].strip()

        if text and data_ocr['highlighted'][i]:
            word_list.append(text + " ")

    # concat all words into one string
    word_string = "".join(word_list)
    # repalce consecutive newlines with one newline
    word_string = re.sub(r'\n+', '\n', word_string).strip()

    return word_string


def main(args):
    img_input = str(args.img_input)

    # Load image
    img_orig = cv2.imread(img_input)

    # Grayscale and apply Otsu's threshold
    img_thresh = threshold_image(img_orig)

    # ocr
    data_ocr = pytesseract.image_to_data(
        img_thresh, lang='eng', config='--psm 6', output_type=Output.DICT)

    # print(data_ocr)
    # for i in range(len(data_ocr['text'])):
    #     word = Word(data_ocr['level'][i], data_ocr['page_num'][i], data_ocr['block_num'][i], data_ocr['par_num'][i], data_ocr['line_num'][i],
    #                 data_ocr['word_num'][i], data_ocr['top'][i], data_ocr['left'][i], data_ocr['width'][i], data_ocr['height'][i], data_ocr['conf'][i], data_ocr['text'][i])
    #     print("Level: {}; Page: {}; Block: {}; Paragraph: {}; Line: {}; Word: {}; Text: {}".format(
    #         word.level, word.page_num, word.block_num, word.par_num, word.line_num, word.word_num, word.text))

    string_ocr = pytesseract.image_to_string(
        img_thresh, lang='eng', config='--psm 6')
    print("Start")

    print(string_ocr)
    print("End")

    # yellow highlight colour range
    hsv_lower = [22, 30, 30]
    hsv_upper = [45, 255, 255]

    img_mask, contours, hierachy = mask_image(
        img_orig, hsv_lower, hsv_upper)

    data_ocr = find_highlighted_words(
        img_mask, data_ocr, threshold_percentage=25)

    # draw all ocr rect
    img_orig_all_ocr = mark_all_words(img_orig.copy(), data_ocr)

    img_orig_rects = draw_contour_rectangles(
        img_mask, img_orig.copy(), contours)

    img_orig_ocr = mark_highlighted_words(
        img_orig.copy(), data_ocr)

    str_highlight = words_to_string(data_ocr)
    print("Start")
    print(str_highlight)
    print("End")

    # # find connected components
    # contours, hierarchy, = cv2.findContours(
    #     img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # # draw only highlighted ocr rects
    # img_ocr_highlight = img_orig.copy()
    # for i in range(n_boxes):
    #     (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
    #                     [i], data_ocr['width'][i], data_ocr['height'][i])
    #     rect_word = Rectangle(x, y, x+w, y+h)
    #     area_word = float((rect_word.xmax - rect_word.xmin)
    #                       * (rect_word.ymax - rect_word.ymin))
    #     for c in contours:
    #         x, y, w, h = cv2.boundingRect(c)
    #         rect_contour = Rectangle(x, y, x+w, y+h)
    #         area_intersect = intersect_area(rect_contour, rect_word)
    #         percent = area_intersect / area_word

    #         if (percent >= 0.5):
    #             cv2.rectangle(img_ocr_highlight, (rect_word.xmin, rect_word.ymin),
    #                           (rect_word.xmax, rect_word.ymax), (0, 255, 0), 2)

    # for c in contours:
    #     x, y, w, h = cv2.boundingRect(c)
    #     rect_contour = Rectangle(x, y, x+w, y+h)
    #     cv2.rectangle(img_mask_highlight, (rect_contour.xmin, rect_contour.ymin),
    #                   (rect_contour.xmax, rect_contour.ymax), (0, 255, 0), 2)

    # cv2.drawContours(image=img_mask_highlight, contours=contours, contourIdx=-1,
    #                  color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    # box_w = 5
    # box_h = 5
    # threshold_perc = 25
    # threshold = (box_w*box_h*threshold_perc)/100

    # # ccomp
    # img_mask_highlight = img_orig.copy()
    # img_mask_contour_filled = img_mask.copy()
    # contours, hierarchy, = cv2.findContours(
    #     img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.drawContours(img_mask_contour_filled, contours, -1,
    # #                  (255, 255, 255), cv2.FILLED, 8, hierarchy)
    # for idx, c in enumerate(contours):
    #     # if(hierarchy[0][idx][3] != -1):  # Discard contours that are holes
    #     #     continue
    #     cv2.drawContours(img_mask_contour_filled, contours, idx,
    #                      (255, 255, 255), cv2.FILLED, 8, hierarchy)
    #     xmin, ymin, w, h = cv2.boundingRect(c)
    #     xmax = xmin + w
    #     ymax = ymin + h

    #     # Scan the image with in bounding box
    #     for x in range(xmin, xmax, box_w):
    #         for y in range(ymin, ymax, box_h):
    #             rect_roi = Rectangle(x, y, x+box_w, y+box_h)
    #             roi = img_mask_contour_filled[y:y+box_h, x:x+box_w]
    #             count = cv2.countNonZero(roi)

    #             if count > threshold:
    #                 cv2.rectangle(img_mask_highlight, (rect_roi.xmin, rect_roi.ymin),
    #                               (rect_roi.xmax, rect_roi.ymax),
    #                               (255, 0, 0), 1, 8, 0)

    # # external
    # img_mask_highlight2 = img_orig.copy()
    # img_mask_contour_filled2 = img_mask.copy()
    # contours, hierarchy, = cv2.findContours(
    #     img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.drawContours(img_mask_contour_filled2, contours, -1,
    # #                  (255, 255, 255), cv2.FILLED, 8, hierarchy)
    # for idx, c in enumerate(contours):
    #     cv2.drawContours(img_mask_contour_filled2, contours, idx,
    #                      (255, 255, 255), cv2.FILLED, 8, hierarchy)
    #     xmin, ymin, w, h = cv2.boundingRect(c)
    #     xmax = xmin + w
    #     ymax = ymin + h

    #     # Scan the image with in bounding box
    #     for x in range(xmin, xmax, box_w):
    #         for y in range(ymin, ymax, box_h):
    #             rect_roi = Rectangle(x, y, x+box_w, y+box_h)
    #             roi = img_mask_contour_filled2[y:y+box_h, x:x+box_w]
    #             count = cv2.countNonZero(roi)

    #             if count > threshold:
    #                 cv2.rectangle(img_mask_highlight2, (rect_roi.xmin, rect_roi.ymin),
    #                               (rect_roi.xmax, rect_roi.ymax),
    #                               (255, 0, 0), 1, 8, 0)

    # stack images
    img_ocr_row = np.concatenate(normalize_images(
        (
            img_orig,
            # img_orig,
            img_thresh,
            # output,
            img_orig_all_ocr,
        )), axis=1)

    # img_ocr_row = vstack((img_orig, img_thresh, img_ocr))

    img_contour_row = np.concatenate(normalize_images(
        (
            img_mask,
            img_orig_rects,
            img_orig_ocr,
            # img_orig_ocr,
            # img_orig_ocr
            # img_mask_contour_filled3,
            # img_mask_highlight3,
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
