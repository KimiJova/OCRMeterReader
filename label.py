import pandas as pd
import re
import cv2
from easyocr import Reader
import os
from tqdm import tqdm

# Suppose `actual_values.csv` has two columns: 'Image_ID' and 'Actual_Value'
actual_df = pd.read_csv('/csv/Readings.csv')    

# Dictionary to store extracted digits for each image
extracted_data = {}

# Specify the folder path
folder_path = '/images/'

# Get all file names in the folder
file_names = os.listdir(folder_path)

file_names = [folder_path + s for s in file_names]

# Sort file names by extracting the number from each file name
sorted_file_names = sorted(file_names, key=lambda x: int(re.search(r'\d+', x).group()))

# Convert to a list of file names (no string formatting)
image_files = sorted_file_names

reader = Reader(['en'], gpu=True)

# OCR function using EasyOCR (assuming `reader` is initialized as in the previous example)
def extract_digits_from_image(image_path):
    results = reader.readtext(image_path)
    digit_string = ""
    for (bbox, text, prob) in tqdm(results):
        digits = re.findall(r'\d+', text)
        if digits:
            digit_string += ''.join(digits)
    return digit_string

# Extract digits from each image and store in dictionary
for image_file in image_files:
    extracted_data[image_file] = extract_digits_from_image(image_file)

# Convert extracted data to a DataFrame
extracted_df = pd.DataFrame(list(extracted_data.items()), columns=['id', 'Extracted_Value'])

combined_df = pd.concat([actual_df, extracted_df], axis=1)

combined_df.head()