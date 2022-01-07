# https://stackoverflow.com/a/60011747/1967693
import cv2
import pytesseract
import numpy as np

# Load image, grayscale, Otsu's threshold
image = cv2.imread('data/book.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(
    gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Draw bounding boxes
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)

# OCR
data = pytesseract.image_to_string(255 - thresh, lang='eng', config='--psm 6')
print(data)

# cv2.imshow('thresh', thresh)
cv2.imshow('image', image)
cv2.waitKey()
