# model.py
"""
Train an emotion detector using HOG features + SVM.
Input: images in data/ (assumes either fer2013.csv OR image folders)
Output: models/emotion_svm.pkl
"""

import os
import numpy as np # type: ignore
import joblib # type: ignore
from skimage.feature import hog # type: ignore
from skimage import color, exposure, io, transform # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.svm import LinearSVC # type: ignore
from sklearn.metrics import classification_report, accuracy_score # type: ignore
from glob import glob
from tqdm import tqdm # type: ignore

# ---- Config ----
IMAGE_SIZE = (48, 48)  # FER2013 style
MODEL_OUT = "models/emotion_svm.pkl"
DATA_DIR = "data"  # Put image folders or fer2013.csv here
CLASSES = None  # will auto-detect if using image folders

# ---- Helper functions ----
def load_images_from_folders(data_dir):
    """
    Expect data_dir structure:
      data/train/<label>/*.jpg
      data/val/<label>/*.jpg   (optional)
    Or just data/<label>/*.jpg
    """
    X = []
    y = []
    # If there's a train/val split folders, use them; else just search data/*
    search_base = [os.path.join(data_dir, "*")] 
    for label_folder in glob(os.path.join(data_dir, "*")):
        if os.path.isdir(label_folder):
            label = os.path.basename(label_folder)
            for f in glob(os.path.join(label_folder, "*.*")):
                try:
                    img = io.imread(f)
                    # convert to gray and resize
                    if img.ndim == 3:
                        img = color.rgb2gray(img)
                    img = transform.resize(img, IMAGE_SIZE, anti_aliasing=True)
                    X.append(img)
                    y.append(label)
                except Exception as e:
                    print("Failed to read", f, e)
    return np.array(X), np.array(y)

def extract_hog_features(images):
    feats = []
    for img in images:
        # skimage hog expects grayscale image scaled 0..1
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), block_norm='L2-Hys')
        feats.append(fd)
    return np.array(feats)

# ---- Main ----
def main():
    # 1) Load images
    print("Loading images from", DATA_DIR)
    X, y = load_images_from_folders(DATA_DIR)
    if len(X) == 0:
        raise SystemExit("No images found in data/. Place images under data/<label>/*.jpg")

    print("Found", len(X), "images across", len(np.unique(y)), "labels:", np.unique(y))
    # 2) Extract HOG features
    print("Extracting HOG features...")
    X_feats = extract_hog_features(X)

    # 3) split
    X_train, X_test, y_train, y_test = train_test_split(X_feats, y, test_size=0.15, random_state=42, stratify=y)

    # 4) train SVM
    print("Training Linear SVM (this may take a bit)...")
    clf = LinearSVC(max_iter=10000)
    clf.fit(X_train, y_train)

    # 5) evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # 6) save model and label mapping
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": clf}, MODEL_OUT)
    print("Saved model to", MODEL_OUT)

if __name__ == "__main__":
    main()
