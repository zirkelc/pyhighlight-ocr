try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

# If you don't have tesseract executable in your PATH, include the following:
#pytesseract.pytesseract.tesseract_cmd = r'<full_path_to_your_tesseract_executable>'
# Example tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'

# Simple image to string
print(pytesseract.image_to_string(Image.open('data/test.png')))

# In order to bypass the image conversions of pytesseract, just use relative or absolute image path
# NOTE: In this case you should provide tesseract supported images or tesseract will return error
print(pytesseract.image_to_string('data/test.png'))

# List of available languages
print(pytesseract.get_languages(config=''))

# French text image to string
# print(pytesseract.image_to_string(Image.open('data/test-european.jpg'), lang='fra'))

# Batch processing with a single file containing the list of multiple image file paths
# print(pytesseract.image_to_string('data/images.txt'))

# Timeout/terminate the tesseract job after a period of time
try:
    # Timeout after 2 seconds
    print(pytesseract.image_to_string('data/test.jpg', timeout=2))
    # Timeout after half a second
    print(pytesseract.image_to_string('data/test.jpg', timeout=0.5))
except RuntimeError as timeout_error:
    # Tesseract processing is terminated
    pass

# Get bounding box estimates
print(pytesseract.image_to_boxes(Image.open('data/test.png')))

# Get verbose data including boxes, confidences, line and page numbers
print(pytesseract.image_to_data(Image.open('data/test.png')))

# Get information about orientation and script detection
# print(pytesseract.image_to_osd(Image.open('data/test.png')))

# Get a searchable PDF
pdf = pytesseract.image_to_pdf_or_hocr('data/test.png', extension='pdf')
with open('data/test.pdf', 'w+b') as f:
    f.write(pdf)  # pdf type is bytes by default

# Get HOCR output
hocr = pytesseract.image_to_pdf_or_hocr('data/test.png', extension='hocr')

# Get ALTO XML output
xml = pytesseract.image_to_alto_xml('data/test.png')
