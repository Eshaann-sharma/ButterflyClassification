# classify_by_color.py
import cv2
import os
import numpy as np
import pandas as pd

def get_dominant_color(image, k=4):
    img = image.reshape((-1, 3))
    img = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = centers[np.argmax(counts)]
    return [int(x) for x in dominant]

def closest_color_name(rgb_tuple):
    b, g, r = rgb_tuple
    if r > 200 and g > 200 and b < 100:
        return 'Yellow'
    elif b > 200 and r < 100 and g < 100:
        return 'Blue'
    elif g > 200 and r < 100 and b < 100:
        return 'Green'
    elif r > 200 and g < 100 and b < 100:
        return 'Red'
    elif r > 200 and g > 100 and b < 100:
        return 'Orange'
    elif r > 200 and g < 100 and b > 100:
        return 'Pink'
    elif r > 150 and g > 150 and b > 150:
        return 'White'
    elif r < 50 and g < 50 and b < 50:
        return 'Black'
    elif r > 100 and g > 60 and b < 50:
        return 'Brown'
    else:
        return 'Other'

def classify_and_save(csv_file, label_file, img_folder, out_csv):
    df = pd.read_csv(csv_file)
    label_map = pd.read_csv(label_file).set_index('filename')['label'].to_dict()
    results = []
    for fname in df['filename']:
        img_path = os.path.join(img_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Missing: {img_path}")
            continue
        dominant = get_dominant_color(img)
        color_name = closest_color_name(dominant)
        label = label_map.get(fname, "Unknown")
        results.append({'filename': fname, 'label': label, 'primary_color': color_name})
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")

if __name__ == "__main__":
    csv_file = "archive-2/Training_set.csv"    # Same CSV for scan, or Testing_set.csv
    label_file = "archive-2/Training_set.csv"
    img_folder = "processed"
    out_csv = "color_results.csv"
    classify_and_save(csv_file, label_file, img_folder, out_csv)
