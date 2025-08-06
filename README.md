# 🦋 Butterfly Color Classification

Classify butterfly images based on their **dominant wing color** using image processing techniques and color space transformations. This project uses a combination of preprocessing, segmentation (GrabCut), KMeans clustering, and LAB color distance for robust and explainable color tagging.

---

## Features

-  Foreground extraction using **GrabCut**
-  Color-based segmentation using **KMeans**
-  Conversion to **LAB color space** for accurate color comparison
-  Weighted LAB distance for robust **dominant color prediction**
-  Optional debug mode to visualize classification results

---

## Color Categories

The system supports the following standard color classes:

`Red`, `Green`, `Blue`, `Yellow`, `Orange`, `Pink`, `Brown`, `White`, `Black`, `Gray`, `Purple`, `Other`

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Eshaann-sharma/ButterflyClassification.git
cd ButterflyClassification
```

### 2.  Install dependencies

```bash
pip install opencv-python numpy pandas scikit-image
```

## Data Cleaning & Preprocessing

- All images are resized to a **fixed resolution** suitable for model input.
- Pixel values are **normalized to [0,1]** for stable neural network training.
- Textual labels are **encoded into integer values** for classification.
- Corrupted or unreadable image files are automatically detected and skipped.

### Before And After preprocessing :

<img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/549ae2c4-8373-470e-b5fb-fca71c726f19" />



<img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/bed22d33-9e7d-4e1b-8e88-40bb64f21dcd" />


---

## Model Training

- A **lightweight convolutional neural network (CNN)** is used for fast training, optimized for low-resource environments.
- The model is compiled and trained using a **minimal runtime setup**.
- Even though peak accuracy is not the main goal, the model captures significant visual features for practical classification.

## Result

<img width="641" height="539" alt="Screenshot 2025-08-07 at 1 54 29 AM" src="https://github.com/user-attachments/assets/d0e97d6e-9bea-4abd-b542-dab20b29801a" />

