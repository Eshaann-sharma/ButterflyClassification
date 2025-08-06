import cv2
import numpy as np
import os
import pandas as pd

def enhance_image(input_path, output_path, size=(224, 224)):
    img = cv2.imread(input_path)
    if img is None:
        return False

    # Resize to standard size
    img_resized = cv2.resize(img, size)

    # Convert to HSV for contrast enhancement
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

    # Histogram equalization on the V (brightness) channel
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    # Morphological opening to reduce small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_clean = cv2.morphologyEx(img_eq, cv2.MORPH_OPEN, kernel)

    # Denoising
    img_denoised = cv2.fastNlMeansDenoisingColored(img_clean, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    # Save preprocessed image
    cv2.imwrite(output_path, img_denoised)
    return True

def batch_preprocess_images(csv_file, img_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_file)
     
    c=0 
    
    for fname in df['filename']:
        in_path = os.path.join(img_folder, fname)
        out_path = os.path.join(output_folder, fname)
        success = enhance_image(in_path, out_path)
        if not success:
            print(f"[ERROR] Could not process image: {in_path}")
        else:
            print(c)
            c=c+1

if __name__ == "__main__":
    
    csv_file = "archive-2/Training_set.csv"
    img_folder = "archive-2/train"
    output_folder = "processed"
    batch_preprocess_images(csv_file, img_folder, output_folder)
