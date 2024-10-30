import cv2
import pytesseract
import matplotlib.pyplot as plt
from easyocr import Reader

def preprocess_and_ocr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    
    # Apply different preprocessing techniques
    img_median = cv2.medianBlur(img, 5)
    
    ret, th1 = cv2.threshold(img_median, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img_median, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 2)
    th3 = cv2.adaptiveThreshold(img_median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 2)
    
    # Store images and titles
    titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    
    # Perform OCR on each preprocessed image and store results
    ocr_results = []
    reader = Reader(['en'], gpu=False)
    for idx, i in enumerate(images):
        results = reader.readtext(i)
        ocr_results.append(results[idx])
    
    return ocr_results, images, titles

# Example usage
image_path = 'images/file_0.jpg'
ocr_results, images, titles = preprocess_and_ocr(image_path)

# Display the results
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

print("OCR Results:")
for result in ocr_results:
    print(result)
    print('\n')