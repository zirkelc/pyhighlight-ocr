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
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from collections import namedtuple
import re

Rectangle = namedtuple('Rectangle', ['xmin', 'ymin', 'xmax', 'ymax'])
# Word = namedtuple('Word', ['level', 'page_num',
#                   'block_num', 'par_num', 'line_num', 'word_num', 'top', 'left', 'width', 'height', 'conf', 'text'])

# https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles


class Levels:
    """Hierarchy levels Tesseract tags each entry in its output data with.

    An entry is either one of the four layout markers or an actual word, so the
    word entries have to be picked out by level before their boxes are used.
    """
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
    """Grayscale image and apply Otsu's threshold.

    Tesseract works best on clean black-on-white input, so the photo is reduced
    to two tones. Otsu's method is used instead of a fixed cutoff because it
    derives the threshold per image, by picking the value that best separates
    the two peaks of the brightness histogram. That adapts to whatever lighting
    the page was photographed under. Yellow highlighter is bright enough to fall
    on the white side of the split, so highlighted text survives as black ink on
    white and stays readable to the OCR.
    """
    # Grayscale
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    # Binarisation and Otsu's threshold
    _, img_thresh = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_thresh, img_gray


def mask_image(img_src, lower, upper):
    """Convert image from RGB to HSV and create a mask for given lower and upper boundaries.

    In RGB there is no single channel that means "yellow": it is a ratio between
    channels, and all three numbers shift with the lighting, so no fixed range
    can catch a highlight both in shadow and under glare. HSV splits the pixel
    into hue (which color), saturation (how vivid) and value (how bright), which
    isolates color from lighting. Dimming the light drops the value but leaves
    the hue where it is, so one narrow hue band matches the same highlighter
    across an unevenly lit page.

    Beware that OpenCV stores hue on a 0-179 scale rather than 0-360, so it fits
    in a uint8. The caller's yellow band of 22-45 is therefore roughly 44 to 90
    degrees on a normal color wheel.

    The saturation and value floors carry their own weight. White paper has no
    meaningful hue, but its hue is still some number that may land inside the
    yellow band by accident, so it is the saturation floor that actually rejects
    the page. The value floor likewise rejects near-black pixels (deep shadow and
    ink). inRange demands all three channels pass and returns a single-channel
    binary mask: white where the pixel was yellow, black everywhere else.

    Note that yellow is an easy case because it sits in the middle of the hue
    range. A red highlighter straddles the wrap-around point (both near 0 and
    near 179) and would need two ranges OR'd together.
    """
    # RGB to HSV color space conversion
    img_hsv = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)
    hsv_lower = np.array(lower, np.uint8)  # Lower HSV value
    hsv_upper = np.array(upper, np.uint8)  # Upper HSV value

    # Color segmentation with lower and upper threshold ranges to obtain a binary image
    img_mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)

    return img_mask, img_hsv

def apply_mask(img_src, img_mask):
    """Apply bitwise conjunction of source image and image mask.

    Keeps the source pixels wherever the mask is white and blacks out the rest.
    Purely illustrative: the detection logic only ever reads the binary mask.
    """

    img_result = cv2.bitwise_and(img_src, img_src, mask=img_mask)

    return img_result


def denoise_image(img_src):
    """Denoise image with a morphological transformation.

    The raw mask is speckled: JPEG artifacts, warm-tinted paper fibers and stray
    pixels along letter edges all sneak past the color range test. An opening is
    an erosion followed by a dilation. The erosion shaves a border off every
    white region, which annihilates anything thinner than the kernel (the
    speckles vanish outright), and the dilation grows the survivors back to
    roughly their original size. Small noise is deleted, large highlight blobs
    come back essentially unchanged.
    """

    # Morphological transformations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
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

def draw_word_boundings(img_src, data_ocr, highlighted_words=False):
    """Draw word bounding boxes"""

    # Convert source image to RGB if it's grayscaled
    img_result = cv2.cvtColor(img_src, cv2.COLOR_GRAY2BGR) if img_src.ndim == 2 else img_src.copy()

    # Iterate through all words
    for i in range(len(data_ocr['text'])):
        # Skip for all non-word elements
        if data_ocr['level'][i] != Levels.WORD:
            continue
        if highlighted_words and not data_ocr['highlighted'][i]:
            continue
        # Get bounding box position and size of word
        (x, y, w, h) = (data_ocr['left'][i], data_ocr['top']
                        [i], data_ocr['width'][i], data_ocr['height'][i])
        # Draw bounding box in red                
        cv2.rectangle(img_result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return img_result


def draw_contour_boundings(img_src, img_mask, threshold_area=400):
    """Draw contour bounding and contour bounding box"""
    # Contour detection
    contours, hierarchy, = cv2.findContours(
        img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create two copies of source image
    img_contour = img_src.copy()
    img_box = img_src.copy()

    for idx, c in enumerate(contours):
        # Skip small contours because its probably noise
        if  cv2.contourArea(c) < threshold_area:
            continue

        # Draw contour in red
        cv2.drawContours(img_contour, contours, idx, (0, 0, 255), 2, cv2.LINE_4, hierarchy)

        # Get bounding box position and size of contour
        x, y, w, h = cv2.boundingRect(c)
        # Draw bounding box in blue
        cv2.rectangle(img_box, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA, 0)

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
    """Find highlighted words by calculating how much of the words area contains white pixels compared to balack pixels.

    This is where the two independent passes meet. The OCR knows where the words
    are but never saw a color; the mask knows where the yellow is but never saw a
    letter. Their only contract is pixel coordinates: for each word, crop that
    word's bounding box out of the mask and count how much of it came back white.

    The area threshold matters. A highlighted word is never fully yellow, because
    the letters themselves are dark ink that the color range rejects, so they
    punch word-shaped holes into the mask (the commented-out contour fill in
    denoise_image is an alternative fix for the same problem, flooding each blob
    solid). The question is therefore not "is this box yellow" but "is a
    meaningful fraction of it yellow", which also keeps words that merely brush
    against a stroke from being counted.
    """

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


def draw_text(
    img,
    *,
    text,
    uv_top_left,
    color=(255, 255, 255),
    fontScale=0.5,
    thickness=1,
    fontFace=cv2.FONT_HERSHEY_COMPLEX,
    outline_color=(0, 0, 0),
    line_spacing=1.5,
):
    """
    Draws multiline with an outline.
    https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    """
    assert isinstance(text, str)

    uv_top_left = np.array(uv_top_left, dtype=float)
    assert uv_top_left.shape == (2,)

    for line in text.splitlines():
        (w, h), _ = cv2.getTextSize(
            text=line,
            fontFace=fontFace,
            fontScale=fontScale,
            thickness=thickness,
        )
        uv_bottom_left_i = uv_top_left + [0, h]
        org = tuple(uv_bottom_left_i.astype(int))

        if outline_color is not None:
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=outline_color,
                thickness=thickness * 3,
                lineType=cv2.LINE_AA,
            )
        cv2.putText(
            img,
            text=line,
            org=org,
            fontFace=fontFace,
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

        uv_top_left += [0, h * line_spacing]

def draw_separation_lines(img_src, images=1):
    height, width = img_src.shape[:2]
    img_result = img_src.copy()

    for i in range(1, images):
        x = int((width / images) * i)
        img_result = cv2.line(img_result, (x, 0), (x, height), (255, 255, 255), thickness=2)

    return img_result

def words_to_string(data_ocr):
    """Concatenate the highlighted words back into readable text.

    Tesseract reports the page, block, paragraph and line markers alongside the
    words themselves, so the original layout can be rebuilt by emitting a newline
    whenever one of those markers goes by. Without it the result would be a flat
    bag of words. Most markers separate text that was not highlighted, which is
    why the runs of newlines they leave behind get collapsed at the end.
    """
    word_list = []
    line_breaks = (Levels.PAGE, Levels.BLOCK, Levels.PARAGRAPH, Levels.LINE)

    for i in range(len(data_ocr['text'])):
        # print("Level: {}; Page: {}; Block: {}; Paragraph: {}; Line: {}; Word: {}; Highlighted: {} Text: {}".format(
        #     data_ocr['level'][i],
        #     data_ocr['page_num'][i],
        #     data_ocr['block_num'][i],
        #     data_ocr['par_num'][i],
        #     data_ocr['line_num'][i],
        #     data_ocr['word_num'][i],
        #     data_ocr['highlighted'][i],
        #     data_ocr['text'][i]))

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
    """Run the OCR and return positional data rather than plain text.

    Output.DICT is what makes the whole approach possible: instead of a bare
    string, Tesseract returns parallel arrays holding each item's level, text and
    bounding box, and those boxes are what the color mask is later sampled
    through. Page segmentation mode 6 assumes a single uniform block of text,
    which suits a book page and stops Tesseract from hunting for columns or other
    layout structure.
    """
    return pytesseract.image_to_data(
        img_src, lang='eng', config='--psm 6', output_type=Output.DICT)


def image_to_string(img_src):
    return pytesseract.image_to_string(
        img_src, lang='eng', config='--psm 6')


def main(args):
    """Extract the highlighted text from a photo of a book page.

    Two passes run over the same photo and never see each other's work. One reads
    the thresholded image and finds where the words are, without any notion of
    color. The other reads the original color image and finds where the yellow
    is, without any notion of text. Joining them by pixel coordinates is what
    identifies a highlighted word. Everything after that is drawing figures.
    """
    img_input = str(args.img_input)

    # Load image
    img_orig = cv2.imread(img_input)

    # First pass: find where the words are
    # Grayscale and apply Otsu's threshold
    img_thresh, img_gray = threshold_image(img_orig)

    # get ocr data
    data_ocr = image_to_data(img_thresh)

    # Second pass: find where the yellow is
    # yellow highlight colour range, on OpenCV's 0-179 hue scale
    hsv_lower = [22, 30, 30]
    hsv_upper = [45, 255, 255]

    # Color segmentation, on the original image because the thresholded one has
    # no color left to segment
    img_mask, img_hsv = mask_image(
        img_orig, hsv_lower, hsv_upper)

    # Noise reduction
    img_mask_denoised = denoise_image(
        img_mask)

    # Apply mask on original image
    img_orig_masked = apply_mask(img_orig, img_mask=img_mask_denoised)

    # Apply mask on thresholded image
    img_thresh_masked = apply_mask(img_thresh, img_mask=img_mask_denoised)

    # Join the two passes: sample the color mask through the OCR bounding boxes
    data_ocr = find_highlighted_words(
        img_mask_denoised, data_ocr, threshold_percentage=25)

    # Everything below is presentation: each stage gets stacked side by side,
    # shown in its own window and written to output/
    # Draw contour boundings and bounding boxes
    img_orig_bounding_contour, img_orig_bounding_box = draw_contour_boundings(img_orig, img_mask=img_mask_denoised)

    # Draw word boundings for highlighted words
    img_thresh_word_boundings = draw_word_boundings(img_thresh, data_ocr, highlighted_words=True)
    img_mask_word_boundings = draw_word_boundings(img_mask_denoised, data_ocr, highlighted_words=True)
    img_orig_word_boundings = draw_word_boundings(img_orig, data_ocr, highlighted_words=True)

    # draw all ocr rect
    img_orig_all_ocr = draw_word_boundings(img_orig.copy(), data_ocr)
    img_thresh_all_ocr = draw_word_boundings(img_thresh.copy(), data_ocr)

    img_orig_rects = draw_contour_rectangles(
        img_mask, img_orig.copy())

    img_orig_ocr = mark_highlighted_words(
        img_orig.copy(), data_ocr)


    string_ocr = pytesseract.image_to_string(
        img_thresh, lang='eng', config='--psm 6')
        
    print("\n\n")
    print(string_ocr)
    print("\n\n")

    str_highlight = words_to_string(data_ocr)
    print("\n\n")
    print(str_highlight)
    print("\n\n")

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

    height, width = img_orig.shape[:2]
    img_orig_all_text = np.zeros([height,width,3],dtype=np.uint8)
    img_orig_all_text.fill(255) 
    string_ocr = string_ocr.replace("‘", "'")
    string_ocr = string_ocr.replace("’", "'")
    draw_text(img_orig_all_text, text=string_ocr, color=(0,0,0), outline_color=None, uv_top_left=(100,100), fontScale=1.5,thickness=2)

    img_orig_highlighted_text = np.zeros([height,width,3],dtype=np.uint8)
    img_orig_highlighted_text.fill(255) 
    # string_ocr = string_ocr.replace("‘", "'")
    # string_ocr = string_ocr.replace("’", "'")
    draw_text(img_orig_highlighted_text, text=str_highlight, color=(0,0,0), outline_color=None, uv_top_left=(100,500), fontScale=1.5,thickness=2)


    img_thresholding = np.concatenate(normalize_images(
        (
            img_orig,
            img_gray,
            img_thresh,
        )), axis=1)

    # draw white line between images
    img_thresholding = draw_separation_lines(img_thresholding, images=3)
    cv2.imshow('thresholding', img_thresholding)
    cv2.imwrite("output/thresholding.png", img_thresholding)
    cv2.waitKey(0)

    img_extract_all = np.concatenate(normalize_images(
        (
            img_orig_all_ocr,
            img_orig_all_text,
        )), axis=1)

    img_extract_all = draw_separation_lines(img_extract_all, images=2)
    cv2.imshow('extract_all', img_extract_all)
    cv2.imwrite("output/extract_all.png", img_extract_all)
    cv2.waitKey(0)

    img_color_segmentation = np.concatenate(normalize_images(
        (
            img_orig,
            img_hsv,
            img_mask,
            
        )), axis=1)

    img_color_segmentation = draw_separation_lines(img_color_segmentation, images=3)
    cv2.imshow('img_color_segmentation', img_color_segmentation)
    cv2.imwrite("output/img_color_segmentation.png", img_color_segmentation)
    cv2.waitKey(0)

    img_noise_reduction = np.concatenate(normalize_images(
    (
        img_mask,
        img_mask_denoised
        
    )), axis=1)

    img_noise_reduction = draw_separation_lines(img_noise_reduction, images=2)
    cv2.imshow('img_noise_reduction', img_noise_reduction)
    cv2.imwrite("output/img_noise_reduction.png", img_noise_reduction)
    cv2.waitKey(0)

    img_orig_and_mask = np.concatenate(normalize_images(
    (
        img_orig_masked,
        img_thresh_masked
        
    )), axis=1)

    img_orig_and_mask = draw_separation_lines(img_orig_and_mask, images=2)
    cv2.imshow('img_orig_and_mask', img_orig_and_mask)
    cv2.imwrite("output/img_orig_and_mask.png", img_orig_and_mask)
    cv2.waitKey(0)    

    img_contour_and_bounding = np.concatenate(normalize_images(
    (
        img_orig_bounding_contour,
        img_orig_bounding_box,
        
    )), axis=1)

    img_contour_and_bounding = draw_separation_lines(img_contour_and_bounding, images=2)
    cv2.imshow('img_contour_and_bounding', img_contour_and_bounding)
    cv2.imwrite("output/img_contour_and_bounding.png", img_contour_and_bounding)
    cv2.waitKey(0)  

    img_final = np.concatenate(normalize_images(
    (
        img_mask_word_boundings,
        img_orig_word_boundings,
        img_orig_highlighted_text
        
    )), axis=1)

    img_final = draw_separation_lines(img_final, images=3)
    cv2.imshow('img_final', img_final)
    cv2.imwrite("output/img_final.png", img_final)
    cv2.waitKey(0)  

    img_title = np.concatenate(normalize_images(
    (
        img_orig,
        img_orig_all_ocr,
        img_orig_bounding_contour,
        img_orig_word_boundings
        
    )), axis=1)

    img_title = draw_separation_lines(img_title, images=4)
    cv2.imshow('img_title', img_title)
    cv2.imwrite("output/img_title.png", img_title)
    cv2.waitKey(0) 

if __name__ == "__main__":

    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser(description=__doc__)
    parser.add_argument('img_input', type=Path, help="Input image")
    args = parser.parse_args()
    main(args)
