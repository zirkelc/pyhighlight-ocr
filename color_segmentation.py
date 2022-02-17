import pprint
import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from collections import namedtuple
import re
from matplotlib import pyplot as plt

from argparse import ArgumentParser
from pathlib import Path

from main import normalize_images

parser = ArgumentParser(description=__doc__)
parser.add_argument('img_input', type=Path, help="Input image")
args = parser.parse_args()

img_input = str(args.img_input)

# Load image
img_orig = cv2.imread(img_input)

# yellow highlight colour range
hsv_lower = [22, 30, 30]
hsv_upper = [45, 255, 255]

"""Convert image from RGB to HSV and create a mask for given lower and upper boundaries."""
# RGB to HSV color space conversion
img_hsv = cv2.cvtColor(img_orig, cv2.COLOR_BGR2HSV)
hsv_lower = np.array(hsv_lower, np.uint8)  # Lower HSV value
hsv_upper = np.array(hsv_upper, np.uint8)  # Upper HSV value

# Color segmentation with lower and upper threshold ranges to obtain a binary image
img_mask = cv2.inRange(img_hsv, hsv_lower, hsv_upper)
img_masked = cv2.bitwise_and(img_orig, img_orig, mask=img_mask)

# img_noise = img_mask.copy()
# Morphological transformations to remove small noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
img_noise = cv2.morphologyEx(
		img_mask, cv2.MORPH_OPEN, kernel, iterations=1)


img_contours1 = img_mask.copy()
# Find highligted contour and fill them with white color
contours, hierarchy, = cv2.findContours(
		img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy, = cv2.findContours(
#     img_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for idx, c in enumerate(contours):
		# if(hierarchy[0][idx][3] != -1):  # Discard contours that are holes
		#     continue
		cv2.drawContours(img_contours1, contours, idx,
											(255, 255, 255), cv2.FILLED, 8, hierarchy)

img_contours2 = img_noise.copy()
# Find highligted contour and fill them with white color
contours, hierarchy, = cv2.findContours(
		img_noise, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for idx, c in enumerate(contours):
		cv2.drawContours(img_contours2, contours, idx,
											(255, 255, 255), cv2.FILLED, 8, hierarchy)


img_row = np.concatenate(normalize_images(
		(
				img_orig,
				img_hsv,
				img_mask,
				img_masked,
		)), axis=1)

# img_grid = np.concatenate((img_ocr_row, img_contour_row), axis=0)

cv2.imshow('img', img_row)
cv2.waitKey(0)

# plt.subplot(131),plt.axis("off"),plt.title('original'),plt.imshow(img_orig[...,::-1])
# plt.subplot(132),plt.axis("off"),plt.title('original'),plt.imshow(img_hsv, cmap='bin')
# plt.subplot(133),plt.axis("off"),plt.title('original'),plt.imshow(img_mask[...,::-1])
# plt.show()