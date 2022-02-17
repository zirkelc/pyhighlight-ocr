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
# https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/
# https://realpython.com/python-opencv-color-spaces/
# https://medium.com/globant/maneuvering-color-mask-into-object-detection-fce61bf891d1
import pprint
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from collections import namedtuple
import re
from matplotlib import pyplot as plt

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
# Word = namedtuple('Word', ['level', 'page_num',
#                   'block_num', 'par_num', 'line_num', 'word_num', 'top', 'left', 'width', 'height', 'conf', 'text'])

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


def threshold_image(img_src):
    """Grayscale image and apply Otsu's threshold"""
    # Grayscale
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    # Binary inverse and Otsu's threshold
    img_thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    return img_thresh, img_gray


def mask_image(img_src, lower, upper):
    """Convert image from RGB to HSV and create a mask for given lower and upper boundaries."""
    # RGB to HSV color space conversion
    img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(lower, np.uint8)  # Lower HSV value
    hsv_upper = np.array(upper, np.uint8)  # Upper HSV value

    # Color segmentation with lower and upper threshold ranges to obtain a binary image
    img_mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
    # output = cv2.bitwise_and(img_src, img_src, mask=img_mask)

    return img_mask, img_hsv

def apply_mask(img_src, img_mask):
    """Convert image from RGB to HSV and create a mask for given lower and upper boundaries."""
    img_result = cv2.bitwise_and(img_src, img_src, mask=img_mask)

    return img_result


def denoise_image(img_src):
    """Denoise image with a morphological transformation."""

    # Morphological transformations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_denoise = cv2.morphologyEx(
        img_src, cv2.MORPH_OPEN, kernel, iterations=1)

    # img_contour = img_denoise.copy()

    # # Find highligted contour and fill them with white color    
    # contours, hierarchy, = cv2.findContours(
    #     img_denoise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # contours, hierarchy, = cv2.findContours(
    # #     img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # for idx, c in enumerate(contours):
    #     # if(hierarchy[0][idx][3] != -1):  # Discard contours that are holes
    #     #     continue
    #     cv2.drawContours(img_contour, contours, idx,
    #                      (255, 255, 255), cv2.FILLED, 8, hierarchy)

    return img_denoise #, contours, hierarchy, img_contour

def draw_word_boundings(img_result, data_ocr):
    """Draw word bounding boxes"""

    # Iterate through all words
    for i in range(len(data_ocr['text'])):
        # Skip for all non-word elements
        if data_ocr['level'][i] != Levels.WORD:
            continue
        # Get bounding box position and size of word
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
                        [i], data_ocr['width'][i], data_ocr['height'][i])
        # Draw bounding box in blue                
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


def draw_boundings(img_src, img_mask):
    # Contour detection
    contours, hierarchy, = cv2.findContours(
        img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create two copies of source image
    img_contour = img_src.copy()
    img_box = img_src.copy()

    for idx, c in enumerate(contours):
        # Draw contour in red
        cv2.drawContours(img_contour, contours, idx, (0, 0, 255), 1, cv2.LINE_4, hierarchy)

        # Get bounding box position and size of contour
        x, y, w, h = cv2.boundingRect(c)
        # Draw bounding box in blue
        cv2.rectangle(img_box, (x, y), (x + w, y + h), (255, 0, 0), 1, cv2.LINE_AA, 0)

    return img_contour, img_box

def draw_contour_rectangles(img_contour, img_result, rect_width=10, rect_height=10, threshold_percentage=25):
    """Draw small rectangles within the contour if the respective rectangle area exceeds the defined threshold percentage."""

    # threshold for rectangle area
    threshold = (rect_width * rect_height * threshold_percentage) / 100
    # contours, hierarchy, = cv2.findContours(
    #     img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy, = cv2.findContours(
        img_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                                  (255, 0, 0), 1, cv2.LINE_AA, 0)

    return img_result



def find_highlighted_words(img_mask, data_ocr, threshold_percentage=25):
    # Initiliaze new column for highlight indicator
    data_ocr['highlighted'] = [False] * len(data_ocr['text'])

    for i in range(len(data_ocr['text'])):
        # Get bounding box position and size of word
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
                        [i], data_ocr['width'][i], data_ocr['height'][i])
        # Calculate threshold number of pixels for the area of the bounding box
        rect_threshold = (w * h * threshold_percentage) / 100
        # Select region of interest from image mask
        img_roi = img_mask[y:y+h, x:x+w]
        # Count white pixels in ROI
        count = cv2.countNonZero(img_roi)
        # Set word as highlighted if its white pixels exceeds the threshold value
        if count > rect_threshold:
            data_ocr['highlighted'][i] = True

    return data_ocr


def mark_highlighted_words(img_result, data_ocr):
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
    # repalce multiple consecutive newlines with one single newline
    word_string = re.sub(r'\n+', '\n', word_string).strip()

    return word_string


def image_to_data(img_src):
    return pytesseract.image_to_data(
        img_src, lang='eng', config='--psm 6', output_type=Output.DICT)


def image_to_string(img_src):
    return pytesseract.image_to_string(
        img_src, lang='eng', config='--psm 6')


def main(args):
    img_input = str(args.img_input)

    # Load image
    img_orig = cv2.imread(img_input)

    # Grayscale and apply Otsu's threshold
    img_thresh, img_gray = threshold_image(img_orig)

    # get ocr data
    data_ocr = image_to_data(img_thresh)

    string_ocr = pytesseract.image_to_string(
        img_thresh, lang='eng', config='--psm 6')
    print("Start")
    print(string_ocr)
    print("End")

    # yellow highlight colour range
    hsv_lower = [22, 30, 30]
    hsv_upper = [45, 255, 255]

    # Color segmentation
    img_mask, img_hsv = mask_image(
        img_orig, hsv_lower, hsv_upper)

    # Apply mask on original image
    img_orig_masked = apply_mask(img_orig, img_mask)

    # Noise reduction
    img_noise = denoise_image(
        img_mask)

    data_ocr = find_highlighted_words(
        img_mask, data_ocr, threshold_percentage=25)

    img_orig_bounding_contour, img_orig_bounding_box = draw_boundings(img_orig, img_mask=img_noise)

    img_thresh_word_boundings = draw_word_boundings(cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR), data_ocr)
    img_mask_word_boundings = draw_word_boundings(cv2.cvtColor(img_noise, cv2.COLOR_GRAY2BGR), data_ocr)

    # draw all ocr rect
    img_orig_all_ocr = draw_word_boundings(img_orig.copy(), data_ocr)

    img_orig_rects = draw_contour_rectangles(
        img_mask, img_orig.copy())

    img_orig_ocr = mark_highlighted_words(
        img_orig.copy(), data_ocr)

    str_highlight = words_to_string(data_ocr)
    print("Start")
    print(str_highlight)
    print("End")

    # stack images
    # img_thresholding = np.concatenate(normalize_images(
    #     (
    #         img_orig,
    #         img_gray,
    #         img_thresh,
    #     )), axis=1)

    # img_contour_row = np.concatenate(normalize_images(
    #     (
    #         img_mask,
    #         img_orig_rects,
    #         img_orig_ocr,
    #         # img_orig_ocr,
    #         # img_orig_ocr
    #         # img_mask_contour_filled3,
    #         # img_mask_highlight3,
    #     )), axis=1)

    # img_grid = np.concatenate((img_ocr_row, img_contour_row), axis=0)

    img_thresholding = np.concatenate(normalize_images(
        (
            img_orig,
            img_gray,
            img_thresh,
        )), axis=1)

    cv2.imshow('thresholding', img_thresholding)
    cv2.imwrite("output/thresholding.png", img_thresholding)
    cv2.waitKey(0)

    img_extract_all = np.concatenate(normalize_images(
        (
            img_orig_all_ocr,
            img_orig_all_ocr,
        )), axis=1)

    cv2.imshow('extract_all', img_extract_all)
    cv2.imwrite("output/extract_all.png", img_extract_all)
    cv2.waitKey(0)

    img_color_segmentation = np.concatenate(normalize_images(
        (
            img_orig,
            img_hsv,
            img_mask,
            
        )), axis=1)

    cv2.imshow('img_color_segmentation', img_color_segmentation)
    cv2.imwrite("output/img_color_segmentation.png", img_color_segmentation)
    cv2.waitKey(0)

    img_noise_reduction = np.concatenate(normalize_images(
    (
        img_mask,
        img_noise
        
    )), axis=1)

    cv2.imshow('img_noise_reduction', img_noise_reduction)
    cv2.imwrite("output/img_noise_reduction.png", img_noise_reduction)
    cv2.waitKey(0)

    img_orig_and_mask = np.concatenate(normalize_images(
    (
        img_orig,
        img_mask,
        img_orig_masked
        
    )), axis=1)

    cv2.imshow('img_orig_and_mask', img_orig_and_mask)
    cv2.imwrite("output/img_orig_and_mask.png", img_orig_and_mask)
    cv2.waitKey(0)    

    img_contour_and_bounding = np.concatenate(normalize_images(
    (
        img_orig_bounding_contour,
        img_orig_bounding_box,
        
    )), axis=1)

    cv2.imshow('img_contour_and_bounding', img_contour_and_bounding)
    cv2.imwrite("output/img_contour_and_bounding.png", img_contour_and_bounding)
    cv2.waitKey(0) 

    img_final = np.concatenate(normalize_images(
    (
        img_thresh_word_boundings,
        img_mask_word_boundings,
        
    )), axis=1)

    cv2.imshow('img_final', img_final)
    cv2.imwrite("output/img_final.png", img_final)
    cv2.waitKey(0) 

if __name__ == "__main__":

    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('img_input', type=Path, help="Input image")
    args = parser.parse_args()
    main(args)
