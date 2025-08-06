# preprocess_images.py
import cv2
import os

def preprocess_image(input_path, output_path, size=(224, 224)):
    # Read
    img = cv2.imread(input_path)
    if img is None:
        return False
    # Resize
    img = cv2.resize(img, size)
    
    # Grayscale 
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize (scale 0-1)
    img = img / 255.0
    # Save as .npy for normalized float array, or rescale & save as image
    save_img = (img * 255).astype('uint8')
    cv2.imwrite(output_path, save_img)
    return True

# Example batch runner
def batch_preprocess_images(csv_file, img_folder, output_folder):
    import pandas as pd
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_file)
    for fname in df['filename']:
        in_path = os.path.join(img_folder, fname)
        out_path = os.path.join(output_folder, fname)
        worked = preprocess_image(in_path, out_path)
        if not worked:
            print(f"Error reading {in_path}")

if __name__ == "__main__":
    
    csv_file = "archive-2/Training_set.csv"
    img_folder = "archive-2/train"
    output_folder = "processed"
    batch_preprocess_images(csv_file, img_folder, output_folder)
