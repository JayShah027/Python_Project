# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import zipfile
import gdown
from pathlib import Path

st.set_page_config(page_title="Image Classifier (Streamlit Cloud)")

# CONFIG: use env var or Streamlit secret for the Drive ID / direct download url
# For security, add MODEL_GDRIVE_ID to Streamlit Cloud Secrets (recommended).
GDRIVE_FILE_ID = os.environ.get("1izp4Qrh-FU4pQ_33DyLe33GnsdP7R7dh", "")  # or "1AbCdEf..."  # <--- optional fallback
MODEL_DIR = Path("models/final_saved_model.keras")
LABELS_PATH = Path("models/labels.json")

@st.cache_resource
def ensure_model_available():
    # If model already present locally, do nothing
    if MODEL_DIR.exists() and LABELS_PATH.exists():
        return True

    if not GDRIVE_FILE_ID:
        # If you didn't set a Drive ID and model isn't in repo, fail early
        raise RuntimeError(
            "Model not found in repo and no MODEL_GDRIVE_ID set. "
            "Upload model to a host (Drive/S3/GitHub release) and set MODEL_GDRIVE_ID or commit models/ to repo."
        )

    # Create models dir
    os.makedirs("models", exist_ok=True)

    # Build gdown URL and download archive to models/model.zip
    zip_path = "models/model.zip"
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    st.info("Downloading model (this runs only once at startup)...")
    gdown.download(url, zip_path, quiet=False)

    # Unzip into models/
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("models")

    # cleanup zip
    try:
        os.remove(zip_path)
    except OSError:
        pass

    if not MODEL_DIR.exists() or not LABELS_PATH.exists():
        raise RuntimeError("Downloaded model is missing expected folders/files.")

    return True

@st.cache_resource
def load_model_and_labels():
    ensure_model_available()
    model = tf.keras.models.load_model(str(MODEL_DIR))
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return model, labels

def main():
    st.title("Image Classifier Demo (Streamlit Cloud)")
    try:
        model, labels = load_model_and_labels()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB").resize((224,224))
        st.image(img, caption="Uploaded image", use_column_width=True)
        x = np.array(img)/255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)[0]
        top_idx = int(preds.argmax())
        st.write(f"Prediction: {labels[top_idx]} ({preds[top_idx]*100:.2f}%)")

if __name__ == "__main__":
    main()
