import cv2
import os
import numpy as np
import pandas as pd
from skimage import color

# LAB Centroids for Standard Colors (from sRGB)
color_labels = {
    'Red':    [53.2408, 80.0925, 67.2032],
    'Green':  [87.7347, -86.1827, 83.1793],
    'Blue':   [32.2970, 79.1875, -107.8602],
    'Yellow': [97.1393, -21.5537, 94.4780],
    'Orange': [74.936, 23.929, 78.948],
    'Pink':   [81.2658, 20.0439, -20.5741],
    'Brown':  [37.986, 13.555, 14.059],
    'White':  [100.000, 0.005, -0.010],
    'Black':  [0.000, 0.000, 0.000],
    'Gray':   [53.585, 0.003, -0.006],
    'Purple': [60.320, 98.254, -60.843],
}

def get_dominant_color(image, k=3):
    """Segment butterfly from background using GrabCut and get dominant BGR color."""
    img = cv2.resize(image, (224, 224))
    mask = np.zeros(img.shape[:2], np.uint8)

    # Rectangle for GrabCut (can be tuned)
    rect = (10, 10, 204, 204)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # GrabCut segmentation
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except:
        # Fallback to using whole image
        pixels = img.reshape(-1, 3)
    else:
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        pixels = img[mask2 == 1]
        if len(pixels) < 10:
            pixels = img.reshape(-1, 3)

    # KMeans clustering to get dominant color
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    _, counts = np.unique(labels, return_counts=True)
    dominant = centers[np.argmax(counts)]
    return [int(x) for x in dominant]

def closest_lab_color(bgr):
    """Match a BGR color to the closest named color using weighted LAB distance."""
    rgb = np.asarray([[bgr[::-1]]], dtype=np.uint8)
    lab = color.rgb2lab(rgb / 255.0)[0, 0]

    min_dist = float("inf")
    best_label = 'Other'
    for name, centroid in color_labels.items():
        weights = np.array([0.5, 1.0, 1.0])  # Reduce weight on L (brightness)
        dist = np.sqrt(np.sum(weights * (lab - np.array(centroid)) ** 2))
        if dist < min_dist:
            min_dist = dist
            best_label = name
    return best_label

def classify_and_save(csv_file, label_file, img_folder, out_csv, debug=False):
    """Classify butterflies by dominant color and save results to CSV."""
    df = pd.read_csv(csv_file)
    label_map = pd.read_csv(label_file).set_index('filename')['label'].to_dict()
    results = []

    if debug:
        os.makedirs("debug", exist_ok=True)

    for fname in df['filename']:
        img_path = os.path.join(img_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Skipping missing or unreadable file: {img_path}")
            continue

        dominant = get_dominant_color(img)
        color_name = closest_lab_color(dominant)
        label = label_map.get(fname, "Unknown")
        results.append({
            'filename': fname,
            'label': label,
            'primary_color': color_name
        })

        # Optional: Save debug image with dominant color rectangle
        if debug:
            debug_img = cv2.resize(img.copy(), (224, 224))
            cv2.rectangle(debug_img, (0, 0), (40, 40), tuple(dominant), -1)
            cv2.imwrite(os.path.join("debug", fname), debug_img)

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[INFO] Saved classification results to: {out_csv}")

if __name__ == "__main__":
    csv_file = "archive-2/Training_set.csv"    # Adjust path as needed
    label_file = "archive-2/Training_set.csv"
    img_folder = "processed"
    out_csv = "color_results.csv"

    # Set debug=True to visualize dominant color blocks
    classify_and_save(csv_file, label_file, img_folder, out_csv, debug=False)
