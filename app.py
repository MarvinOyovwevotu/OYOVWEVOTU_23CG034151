# app.py
"""
Simple Streamlit app for emotion detection.
Run: streamlit run app.py
"""
import streamlit as st # type: ignore
from skimage import color, transform, io # type: ignore
from skimage.feature import hog # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
from PIL import Image # type: ignore
import io as sysio

MODEL_PATH = "models/emotion_svm.pkl"
IMAGE_SIZE = (48, 48)

@st.cache_resource
def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"]

def preprocess_image(image_bytes):
    img = Image.open(sysio.BytesIO(image_bytes)).convert("L")  # grayscale
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img) / 255.0
    fd = hog(arr, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), block_norm='L2-Hys')
    return fd.reshape(1, -1)

st.title("Emotion Detector (HOG + SVM) 💫")
st.markdown("Upload a face image (close-up) and I'll predict the emotion.")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
model = None
try:
    model = load_model()
except Exception as e:
    st.error("Model not found. Run `python model.py` to train and create models/emotion_svm.pkl.")
    st.stop()

if uploaded is not None:
    bytes_data = uploaded.read()
    st.image(bytes_data, caption="Uploaded", width=250)
    feats = preprocess_image(bytes_data)
    pred = model.predict(feats)[0]
    st.success(f"Predicted emotion: **{pred}**")
