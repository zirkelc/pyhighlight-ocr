# Python Highlight OCR

Extract highlighted text from book images using Python, OpenCV, and Tesseract OCR.

> [!NOTE]
> I have ported this project to JavScript to run in the browser without any installation. 
> Check out [highlights](https://github.com/zirkelc/highlights).


See my original blog post [Extract highlighted text from a book using Python](https://dev.to/zirkelc/extract-highlighted-text-from-a-book-using-python-e15) for more details.


## Usage

Install the Tesseract OCR engine (`pytesseract` is only a wrapper around it):
```bash
brew install tesseract
```

Install the Python dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run with an image:
```bash
python main.py input/animal1.jpeg
```

This prints the full OCR text and the extracted highlighted text, and opens a series of windows showing each step of the pipeline. Press any key to advance to the next step. Each step is also saved to `output/`.
