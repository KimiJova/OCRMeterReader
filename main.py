from easyocr import Reader
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import re

# PARAMETERS
languages_list = ['en', 'pt']
gpu = True

# Load image with error handling
try:
    img = cv2.imread("images/file_7.jpg")
    if img is None:
        raise ValueError("Image not found or path is incorrect.")
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Copy original image for further processing
original = img.copy()

# Function to apply preprocessing filters
def apply_filter(img):
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # Unsharp mask: sharpen the image
    sharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    # Convert to grayscale
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

# Apply filter to image
#img = apply_filter(img)

# Initialize EasyOCR Reader
reader = Reader(languages_list, gpu=gpu)

# Perform OCR on the preprocessed image
results = reader.readtext(img)

# Helper functions for drawing boxes and text
def box_coordinates(box):
    lt, rt, br, bl = box
    return (int(lt[0]), int(lt[1])), (int(rt[0]), int(rt[1])), (int(br[0]), int(br[1])), (int(bl[0]), int(bl[1]))

def write_text(text, x, y, img, color=(50, 50, 255), font_size=30):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y - font_size), text, fill=color)
    return np.array(img_pil)

def draw_img(img, lt, br, color=(200, 255, 0), thickness=2):
    cv2.rectangle(img, lt, br, color, thickness)
    return img

# Draw OCR results on the image
for (box, text, probability) in results:
    lt, rt, br, bl = box_coordinates(box)
    img = draw_img(img, lt, br)
    img = write_text(text, lt[0], lt[1], img)

# Display the image with bounding boxes and text
cv2.imshow("OCR Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract text and apply regex to find meter values
extracted_text = ' '.join([text[1] for text in results])
print("Extracted Text:", extracted_text)

# Use regex to find all digits in the extracted text
pattern = r'\b\d\b'
meter_values = "".join(re.findall(pattern, extracted_text))
# Print the found digits
print("Extracted Digits:", meter_values)
