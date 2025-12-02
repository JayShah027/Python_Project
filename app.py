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

# Must be called early
st.set_page_config(page_title="Image Classifier (Streamlit Cloud)")

# Use a meaningful env var name. In Streamlit Cloud, set this under Secrets / Environment variables.
GDRIVE_FILE_ID = os.environ.get("1izp4Qrh-FU4pQ_33DyLe33GnsdP7R7dh", "")  # set MODEL_GDRIVE_ID in Streamlit settings
MODEL_DIR = Path("models/final_saved_model.keras")
LABELS_PATH = Path("models/labels.json")


@st.cache_resource
def ensure_model_available():
    """
    Ensure models/ contains a loadable model directory and labels.json.
    If not present locally and an env var is provided, download a zip from Google Drive.
    After extraction we attempt to auto-detect the model folder if its name differs.
    """
    # already present?
    if MODEL_DIR.exists() and LABELS_PATH.exists():
        return True

    # no env var and not in repo => raise
    if not GDRIVE_FILE_ID:
        raise RuntimeError(
            "Model not found in repo and no MODEL_GDRIVE_ID set. "
            "Set MODEL_GDRIVE_ID in Streamlit Cloud or commit a models/ folder to the repo."
        )

    os.makedirs("models", exist_ok=True)
    zip_path = "models/model.zip"

    # Build a drive download URL. gdown accepts the full url or an id with /uc?id=...
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    st.info("Downloading model (this runs only once at startup)...")

    # download (non-quiet so logs show in deployment)
    gdown.download(url, zip_path, quiet=False)

    # extract
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall("models")
    except zipfile.BadZipFile as e:
        # cleanup and re-raise with a helpful message
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise RuntimeError(f"Downloaded file is not a valid zip: {e}")

    # cleanup zip
    try:
        os.remove(zip_path)
    except OSError:
        pass

    # If MODEL_DIR is missing, try to auto-detect a saved model inside models/
    if not MODEL_DIR.exists():
        # candidate types: directory containing saved_model.pb (SavedModel),
        # or folder ending with .keras, or a .h5/.keras file
        for entry in Path("models").iterdir():
            if entry.is_dir():
                # SavedModel detection
                if (entry / "saved_model.pb").exists():
                    MODEL_DIR_resolved = entry
                    break
                # directory named *.keras (tf.keras export)
                if entry.name.endswith(".keras"):
                    MODEL_DIR_resolved = entry
                    break
            else:
                # single file model (e.g., model.h5)
                if entry.suffix in {".h5", ".keras"}:
                    # create a directory wrapper path for load_model
                    MODEL_DIR_resolved = entry
                    break
        else:
            MODEL_DIR_resolved = None

        if MODEL_DIR_resolved:
            # update global path variable in the filesystem (so load_model can use it)
            # Note: we only set MODEL_DIR if a suitable candidate exists
            global MODEL_DIR
            MODEL_DIR = Path(MODEL_DIR_resolved)
        else:
            raise RuntimeError("Downloaded model is missing expected folders/files (no SavedModel/.keras/.h5 found).")

    if not LABELS_PATH.exists():
        # try to find a labels.json anywhere in models/
        found = False
        for p in Path("models").rglob("labels.json"):
            LABELS_PATH_resolved = p
            found = True
            break
        if found:
            global LABELS_PATH
            LABELS_PATH = Path(LABELS_PATH_resolved)
        else:
            raise RuntimeError("labels.json not found inside the downloaded model archive.")

    return True


@st.cache_resource
def load_model_and_labels():
    ensure_model_available()
    # tf.keras.models.load_model accepts a folder (SavedModel) or a file (.h5/.keras)
    model = tf.keras.models.load_model(str(MODEL_DIR))
    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)
    return model, labels


def preprocess_image(img: Image.Image, size=(224, 224)):
    img = img.convert("RGB").resize(size)
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    return img, x


def main():
    st.title("Image Classifier Demo (Streamlit Cloud)")

    # load model
    try:
        with st.spinner("Loading model..."):
            model, labels = load_model_and_labels()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        # provide some hints
        st.caption("Tip: set MODEL_GDRIVE_ID in Streamlit Cloud or commit a models/ folder with final_saved_model.keras and labels.json")
        return

    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if not uploaded:
        st.info("Upload an image to get a prediction.")
        return

    img, x = preprocess_image(Image.open(uploaded))
    st.image(img, caption="Uploaded image", use_column_width=True)

    # prediction
    try:
        preds = model.predict(x)[0]
        top_idx = int(np.argmax(preds))
        st.write(f"Prediction: **{labels[top_idx]}** ({preds[top_idx]*100:.2f}%)")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
